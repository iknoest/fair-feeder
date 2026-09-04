#!/usr/bin/env python3
"""
Fair Feeder Durable Upload Queue & Ledger

Provides durable, per-file upload tracking and verification between local Pi storage and Google Drive.
Guarantees:
1. Every finalized valid video clip is registered as PENDING before upload.
2. Single-file uploads using exact rclone copyto commands (never repeated whole-folder copies).
3. Bounded upload timeout with automatic retry and backoff on transient errors/timeouts.
4. Remote existence and size verification before marking UPLOADED.
5. Idempotent acceptance if remote already matches local size.
6. Survives process crashes and service restarts via persistent atomic ledger.
7. Morning drain gate authority: drain is only satisfied when all required items are UPLOADED.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import fcntl
except ImportError:
    fcntl = None

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
DEFAULT_LEDGER_PATH = repo_root / "upload_ledger.json"
DEFAULT_MAX_WAIT_SEC = 3600  # Bounds one active uploader invocation / attempt cycle
DEFAULT_UPLOAD_TIMEOUT_SEC = 600
DEFAULT_VERIFY_TIMEOUT_SEC = 30
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_MAX_RECOVERY_CYCLES = 3
DEFAULT_RECOVERY_COOLDOWN_SEC = 300.0
DEFAULT_WAKE_TIMER_UNIT = "fair-feeder-uploader-wake"
DEFAULT_WAKE_SERVICE_NAME = "fair-feeder-uploader.service"

from config import (
    LOGITECH_DRIVE_FOLDER_ID,
    CAMERA_TARGETS,
    get_camera_target_for_path,
)


class CorruptLedgerError(RuntimeError):
    """Raised when the upload ledger file is unreadable, corrupt, or has invalid schema."""
    pass


class UploadState:
    PENDING = "PENDING"
    UPLOADING = "UPLOADING"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_EXHAUSTED = "FAILED_EXHAUSTED"
    UPLOADED = "UPLOADED"


def schedule_systemd_wake(
    wait_sec: float,
    service_name: str = DEFAULT_WAKE_SERVICE_NAME,
    timer_unit: str = DEFAULT_WAKE_TIMER_UNIT,
) -> Tuple[bool, str]:
    """
    Schedules a transient systemd timer to wake service_name after wait_sec seconds.
    Uses systemd-run. Bounded and non-hammering: cancels any previous timer for the same unit first.
    """
    sctl = shutil.which("systemctl")
    srun = shutil.which("systemd-run")
    if not sctl or "systemctl" not in sctl or not srun or "systemd-run" not in srun:
        return False, "systemd-run or systemctl not found (non-systemd / local environment)"

    cancel_systemd_wake(timer_unit)

    import math
    delay_sec = max(1, int(math.ceil(wait_sec)))
    cmd = [
        "systemd-run",
        f"--unit={timer_unit}",
        f"--on-active={delay_sec}s",
        "--description=Fair Feeder Autonomous Upload Recovery Wake",
        "systemctl",
        "start",
        service_name,
    ]
    if os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() != 0 and shutil.which("sudo"):
        cmd = ["sudo", "-n"] + cmd

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            msg = f"Scheduled systemd wake timer ({timer_unit}) for {service_name} in {delay_sec}s"
            print(f"⏰ {msg}")
            return True, msg
        else:
            msg = f"Failed to schedule systemd wake timer: {res.stderr.strip()}"
            print(f"⚠️ {msg}")
            return False, msg
    except Exception as e:
        msg = f"Exception scheduling systemd wake timer: {e}"
        print(f"⚠️ {msg}")
        return False, msg


def cancel_systemd_wake(timer_unit: str = DEFAULT_WAKE_TIMER_UNIT) -> Tuple[bool, str]:
    """Cancels any pending systemd transient wake timer."""
    if not shutil.which("systemctl"):
        return False, "systemctl not found"

    cmd = ["systemctl", "stop", f"{timer_unit}.timer"]
    if os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() != 0 and shutil.which("sudo"):
        cmd = ["sudo", "-n"] + cmd

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=False)
        reset_cmd = ["systemctl", "reset-failed", f"{timer_unit}.timer"]
        if os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() != 0 and shutil.which("sudo"):
            reset_cmd = ["sudo", "-n"] + reset_cmd
        subprocess.run(reset_cmd, capture_output=True, text=True, check=False)
        return True, f"Cancelled {timer_unit}.timer"
    except Exception as e:
        return False, str(e)


class UploadQueue:
    """Manages persistent per-file upload state and execution."""

    def __init__(
        self,
        ledger_path: Optional[Path] = None,
        upload_timeout_sec: int = DEFAULT_UPLOAD_TIMEOUT_SEC,
        verify_timeout_sec: int = DEFAULT_VERIFY_TIMEOUT_SEC,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        max_recovery_cycles: int = DEFAULT_MAX_RECOVERY_CYCLES,
        recovery_cooldown_sec: float = DEFAULT_RECOVERY_COOLDOWN_SEC,
        backoff_base_sec: float = 5.0,
        backoff_max_sec: float = 300.0,
    ):
        if ledger_path is None:
            env_path = os.environ.get("UPLOAD_LEDGER_PATH")
            self.ledger_path = Path(env_path) if env_path else DEFAULT_LEDGER_PATH
        else:
            self.ledger_path = Path(ledger_path)

        self.upload_timeout_sec = int(os.environ.get("UPLOAD_QUEUE_UPLOAD_TIMEOUT_SEC", upload_timeout_sec))
        self.verify_timeout_sec = int(os.environ.get("UPLOAD_QUEUE_VERIFY_TIMEOUT_SEC", verify_timeout_sec))
        self.max_attempts = int(os.environ.get("UPLOAD_QUEUE_MAX_ATTEMPTS", max_attempts))
        self.max_recovery_cycles = int(os.environ.get("UPLOAD_QUEUE_MAX_RECOVERY_CYCLES", max_recovery_cycles))
        self.recovery_cooldown_sec = float(os.environ.get("UPLOAD_QUEUE_RECOVERY_COOLDOWN_SEC", recovery_cooldown_sec))
        self.backoff_base_sec = float(os.environ.get("UPLOAD_QUEUE_BACKOFF_BASE_SEC", backoff_base_sec))
        self.backoff_max_sec = float(os.environ.get("UPLOAD_QUEUE_BACKOFF_MAX_SEC", backoff_max_sec))
        self._thread_lock = threading.Lock()
        self._init_ledger()

    def _get_lock_path(self) -> Path:
        return self.ledger_path.with_suffix(".lock")

    def _migrate_v1_to_v2(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrates ledger schema from v1 (filename-only keys) to v2 (camera_type:filename keys).
        Reconciles misclassified USB staging items and sets version=2.
        """
        old_items = data.get("items", {})
        new_items = {}
        for old_key, item in old_items.items():
            filepath_str = item.get("filepath", "")
            target = get_camera_target_for_path(filepath_str)
            expected_cam = target["camera_type"]
            expected_dest = target["rclone_dest_path"]
            expected_remote = target.get("rclone_remote", "gdrive-randomdice:")

            filename = item.get("filename", old_key)

            # Check if historical entry was in USB staging but registered as rtsp or with empty dest
            if expected_cam == "usb":
                cam_type = "usb"
                dest_path = expected_dest
                remote_verified = item.get("remote_verified", False)
                # If it was verified against root (""), it must be re-verified against the USB folder
                if item.get("rclone_dest_path") != expected_dest:
                    remote_verified = False
            else:
                cam_type = item.get("camera_type", "rtsp")
                dest_path = item.get("rclone_dest_path", "")
                remote_verified = item.get("remote_verified", False)

            new_key = f"{cam_type}:{filename}"
            item["key"] = new_key
            item["filename"] = filename
            item["camera_type"] = cam_type
            item["rclone_remote"] = item.get("rclone_remote") or expected_remote
            item["rclone_dest_path"] = dest_path
            item["remote_verified"] = remote_verified
            if not remote_verified and item.get("state") == UploadState.UPLOADED:
                item["state"] = UploadState.PENDING

            if "recovery_cycle" not in item:
                item["recovery_cycle"] = 0
            if "exhausted_at" not in item:
                item["exhausted_at"] = None
            if "exhausted_at_ts" not in item:
                item["exhausted_at_ts"] = None

            new_items[new_key] = item

        data["version"] = 2
        data["items"] = new_items
        return data

    def _init_ledger(self):
        with self._thread_lock:
            if not self.ledger_path.exists():
                self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
                init_data = {
                    "version": 2,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "items": {},
                }
                self._write_ledger_unlocked(init_data)
            else:
                data = self._read_ledger_unlocked()
                if data.get("version") == 1:
                    migrated = self._migrate_v1_to_v2(data)
                    self._write_ledger_unlocked(migrated)

    def _read_ledger_unlocked(self) -> Dict[str, Any]:
        if not self.ledger_path.exists():
            return {"version": 2, "updated_at": None, "items": {}}
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "items" not in data or not isinstance(data.get("items"), dict):
                raise CorruptLedgerError(
                    f"Upload ledger at {self.ledger_path} has invalid schema (expected dict with 'items' dict)"
                )
            if data.get("version") == 1:
                data = self._migrate_v1_to_v2(data)
                try:
                    self._write_ledger_unlocked(data)
                except Exception:
                    pass
            return data
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise CorruptLedgerError(f"Upload ledger at {self.ledger_path} is corrupt: {e}") from e
        except CorruptLedgerError:
            raise
        except Exception as e:
            raise CorruptLedgerError(f"Failed to read upload ledger at {self.ledger_path}: {e}") from e

    def _write_ledger_unlocked(self, data: Dict[str, Any]):
        data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        temp_file = self.ledger_path.with_suffix(".tmp")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        temp_file.replace(self.ledger_path)

    def _locked_transaction(self, callback):
        """Executes a callable within a file lock across processes and threads."""
        lock_path = self._get_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            lock_fd = None
            try:
                if fcntl:
                    lock_fd = open(lock_path, "w")
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
                data = self._read_ledger_unlocked()
                result, new_data = callback(data)
                if new_data is not None:
                    self._write_ledger_unlocked(new_data)
                return result
            finally:
                if lock_fd:
                    try:
                        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
                    try:
                        lock_fd.close()
                    except Exception:
                        pass

    def get_recovery_eligibility(
        self,
        item: Dict[str, Any],
        now: Optional[float] = None,
        cooldown_sec: Optional[float] = None,
        max_recovery_cycles: Optional[int] = None,
    ) -> Tuple[bool, Optional[float]]:
        """
        Determines if a FAILED_EXHAUSTED item is eligible for recovery.
        Single authoritative rule for recovery eligibility.
        Returns: (is_eligible: bool, earliest_eligible_ts: Optional[float])
        - If not FAILED_EXHAUSTED: (False, None)
        - If recovery_cycle >= max_recovery_cycles: (False, None) (permanently exhausted)
        - If exhausted_at_ts is None: (True, current_time)
        - If current_time >= exhausted_at_ts + cooldown_sec: (True, eligible_ts)
        - If cooldown not yet elapsed: (False, eligible_ts)
        """
        if item.get("state") != UploadState.FAILED_EXHAUSTED:
            return False, None

        max_cycles = max_recovery_cycles if max_recovery_cycles is not None else self.max_recovery_cycles
        if item.get("recovery_cycle", 0) >= max_cycles:
            return False, None

        current_time = now if now is not None else time.time()
        cooldown = cooldown_sec if cooldown_sec is not None else self.recovery_cooldown_sec
        exhausted_ts = item.get("exhausted_at_ts")
        if exhausted_ts is None:
            return True, current_time

        eligible_ts = float(exhausted_ts) + cooldown
        if current_time >= eligible_ts:
            return True, eligible_ts
        return False, eligible_ts

    def _apply_recovery_cycle(
        self,
        item: Dict[str, Any],
        max_recovery_cycles: Optional[int] = None,
        reason: str = "after cooldown",
    ):
        """
        Applies a recovery cycle mutation to an eligible item in-place:
        Increments recovery_cycle, resets attempt_count to 0, sets state to PENDING,
        and clears exhaustion markers.
        """
        max_cycles = max_recovery_cycles if max_recovery_cycles is not None else self.max_recovery_cycles
        item["recovery_cycle"] = item.get("recovery_cycle", 0) + 1
        item["attempt_count"] = 0
        item["state"] = UploadState.PENDING
        item["exhausted_at"] = None
        item["exhausted_at_ts"] = None
        item["earliest_recovery_ts"] = None
        item["earliest_recovery_at"] = None
        item["next_attempt_at"] = None
        item["last_error"] = f"Recovered for cycle {item['recovery_cycle']}/{max_cycles} ({reason})"

    def register_file(
        self,
        filepath: Path,
        camera_type: Optional[str] = None,
        rclone_remote: Optional[str] = None,
        rclone_dest_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Durably registers a finalized local file into the ledger as PENDING.
        Must be called BEFORE upload begins.
        Enforces deterministic composite key (camera_type:filename) and immutable upload contract.
        """
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"Cannot register non-existent file for upload: {filepath}")

        file_size = filepath.stat().st_size
        filename = filepath.name

        target = get_camera_target_for_path(filepath)
        effective_camera = camera_type if camera_type is not None else target["camera_type"]
        effective_remote = rclone_remote if rclone_remote is not None else target["rclone_remote"]
        effective_dest = rclone_dest_path if rclone_dest_path is not None else target["rclone_dest_path"]

        key = f"{effective_camera}:{filename}"

        def _tx(data: Dict[str, Any]):
            items = data.setdefault("items", {})
            existing = items.get(key)
            if existing:
                # Enforce immutable contract integrity
                if existing.get("camera_type") != effective_camera:
                    raise ValueError(
                        f"Camera mismatch for ledger key '{key}': existing='{existing.get('camera_type')}', requested='{effective_camera}'"
                    )
                if existing.get("rclone_dest_path") != effective_dest:
                    raise ValueError(
                        f"Destination mismatch for ledger key '{key}': existing='{existing.get('rclone_dest_path')}', requested='{effective_dest}'"
                    )
                if existing.get("rclone_remote") != effective_remote:
                    raise ValueError(
                        f"Remote mismatch for ledger key '{key}': existing='{existing.get('rclone_remote')}', requested='{effective_remote}'"
                    )
                if str(Path(existing.get("filepath", "")).resolve()) != str(filepath):
                    raise ValueError(
                        f"Filepath mismatch for ledger key '{key}': existing='{existing.get('filepath')}', requested='{filepath}'"
                    )

                if existing.get("state") == UploadState.UPLOADED and existing.get("remote_verified"):
                    return existing, None

                # Bounded recovery upon re-registration if exhausted (must NOT bypass cooldown)
                if existing.get("state") == UploadState.FAILED_EXHAUSTED:
                    is_eligible, _ = self.get_recovery_eligibility(existing)
                    if is_eligible:
                        self._apply_recovery_cycle(existing, reason="upon re-registration")
                        return existing, data
                    else:
                        # Cooldown has not elapsed or max recovery cycles reached:
                        # Cannot bypass cooldown; remains FAILED_EXHAUSTED.
                        return existing, None

                return existing, None

            now_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            item = {
                "key": key,
                "filename": filename,
                "filepath": str(filepath),
                "camera_type": effective_camera,
                "rclone_remote": effective_remote,
                "rclone_dest_path": effective_dest,
                "file_size_bytes": file_size,
                "state": UploadState.PENDING,
                "attempt_count": 0,
                "recovery_cycle": 0,
                "exhausted_at": None,
                "exhausted_at_ts": None,
                "created_at": now_iso,
                "last_attempt_at": None,
                "last_error": None,
                "remote_verified": False,
                "uploaded_at": None,
                "next_attempt_at": None,
            }
            items[key] = item
            return item, data

        registered_item = self._locked_transaction(_tx)
        print(f"📋 Registered for durable upload: {key} ({file_size / (1024*1024):.1f} MB, state={registered_item.get('state')})")
        return registered_item

    def get_item(self, key_or_filename: str, camera_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves an item by its exact key (e.g. 'rtsp:motion_...mp4'),
        or by camera-qualified filename, or by unique filename.
        """
        def _tx(data: Dict[str, Any]):
            items = data.get("items", {})
            if key_or_filename in items:
                return items[key_or_filename], None

            if camera_type:
                k = f"{camera_type}:{key_or_filename}"
                if k in items:
                    return items[k], None

            matches = [it for it in items.values() if it.get("filename") == key_or_filename]
            if len(matches) == 1:
                return matches[0], None
            elif len(matches) > 1:
                if camera_type:
                    for m in matches:
                        if m.get("camera_type") == camera_type:
                            return m, None
                raise ValueError(
                    f"Ambiguous item lookup: '{key_or_filename}' exists for multiple cameras: {[m.get('key') for m in matches]}. Specify camera_type."
                )
            return None, None

        return self._locked_transaction(_tx)

    def get_all_items(self) -> Dict[str, Dict[str, Any]]:
        def _tx(data: Dict[str, Any]):
            return dict(data.get("items", {})), None
        return self._locked_transaction(_tx)

    def get_unresolved_items(self) -> List[Dict[str, Any]]:
        """Returns all items in PENDING, UPLOADING, FAILED_RETRYABLE, or FAILED_EXHAUSTED states."""
        def _tx(data: Dict[str, Any]):
            unresolved = [
                item for item in data.get("items", {}).values()
                if item.get("state") in (
                    UploadState.PENDING,
                    UploadState.UPLOADING,
                    UploadState.FAILED_RETRYABLE,
                    UploadState.FAILED_EXHAUSTED,
                )
            ]
            return unresolved, None
        return self._locked_transaction(_tx)

    def has_unresolved_uploads(self) -> Tuple[bool, int, str]:
        """
        Authority check for morning drain gate.
        Returns: (has_unresolved: bool, count: int, summary_description: str)
        """
        unresolved = self.get_unresolved_items()
        if not unresolved:
            return False, 0, "No unresolved uploads in queue"
        names = [u.get("key", u.get("filename")) for u in unresolved[:3]]
        sample_str = ", ".join(names)
        if len(unresolved) > 3:
            sample_str += f" (+{len(unresolved) - 3} more)"
        return True, len(unresolved), f"{len(unresolved)} pending/unresolved upload(s): [{sample_str}]"

    def recover_pending(self) -> int:
        """
        Crash/restart recovery:
        Resets any UPLOADING items to PENDING and validates that local files exist.
        """
        def _tx(data: Dict[str, Any]):
            recovered_count = 0
            items = data.setdefault("items", {})
            for key, item in items.items():
                if item.get("state") == UploadState.UPLOADING:
                    item["state"] = UploadState.PENDING
                    item["last_error"] = "Process restarted while upload in-flight"
                    recovered_count += 1

                if item.get("state") in (UploadState.PENDING, UploadState.FAILED_RETRYABLE):
                    p = Path(item.get("filepath", ""))
                    if not p.exists():
                        item["state"] = UploadState.FAILED_RETRYABLE
                        item["last_error"] = f"Local file missing: {p}"
            return recovered_count, data if recovered_count > 0 else None

        recovered = self._locked_transaction(_tx)
        if recovered > 0:
            print(f"🔄 Recovered {recovered} in-flight upload(s) back to PENDING state.")
        return recovered

    def verify_remote_file(
        self,
        filename: str,
        expected_size: int,
        rclone_remote: str,
        rclone_dest_path: str,
        timeout_sec: Optional[int] = None,
    ) -> Tuple[bool, Optional[int], str]:
        """
        Performs remote verification using cheap `rclone lsf` format.
        Does NOT download the full video.
        """
        timeout = timeout_sec or self.verify_timeout_sec
        rclone_bin = shutil.which("rclone")
        if not rclone_bin:
            return False, None, "rclone binary not available in PATH"

        cmd = [rclone_bin, "lsf", rclone_remote, "--include", filename, "--format", "sp"]
        if len(rclone_dest_path) > 20 and "/" not in rclone_dest_path:
            cmd.extend(["--drive-root-folder-id", rclone_dest_path])
        elif rclone_dest_path:
            cmd[2] = f"{rclone_remote.rstrip('/')}/{rclone_dest_path.lstrip('/')}"

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if res.returncode != 0:
                return False, None, f"rclone lsf returned exit code {res.returncode}: {res.stderr.strip()}"

            for line in res.stdout.strip().splitlines():
                parts = line.split(";", 1)
                if len(parts) == 2:
                    sz_str, name = parts
                    if name.strip() == filename.strip():
                        try:
                            rem_sz = int(sz_str.strip())
                            if rem_sz == expected_size:
                                return True, rem_sz, f"Remote verified: size matches exactly ({rem_sz} bytes)"
                            return False, rem_sz, f"Size mismatch: expected {expected_size}, remote has {rem_sz}"
                        except ValueError:
                            pass
            return False, None, f"File {filename} not found on remote"
        except subprocess.TimeoutExpired:
            return False, None, f"Remote verification timed out after {timeout}s"
        except Exception as e:
            return False, None, f"Remote verification error: {e}"

    def upload_file(self, key_or_filename: str, camera_type: Optional[str] = None) -> Tuple[bool, str]:
        """
        Uploads a single exact file with full lifecycle:
        1. Pre-check idempotency (already on remote).
        2. Set state = UPLOADING.
        3. Run exact rclone copyto.
        4. Verify remote file existence and size.
        5. Set state = UPLOADED on success, or FAILED_RETRYABLE / FAILED_EXHAUSTED on failure.
        """
        item = self.get_item(key_or_filename, camera_type=camera_type)
        if not item:
            return False, f"File not registered in upload ledger: {key_or_filename}"

        key = item.get("key", f"{item.get('camera_type', 'rtsp')}:{item.get('filename')}")
        filename = item["filename"]

        if item.get("state") == UploadState.UPLOADED and item.get("remote_verified"):
            return True, f"Already uploaded and verified: {key}"

        current_attempts = item.get("attempt_count", 0)
        if current_attempts >= self.max_attempts or item.get("state") == UploadState.FAILED_EXHAUSTED:
            msg = f"Max upload attempts ({self.max_attempts}) reached for {key}"
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._update_item_state(
                key,
                UploadState.FAILED_EXHAUSTED,
                error=msg,
                next_attempt_at=None,
                exhausted_at=now_iso,
                exhausted_at_ts=time.time(),
            )
            return False, msg

        local_path = Path(item["filepath"])
        if not local_path.exists():
            new_attempts = current_attempts + 1
            is_exhausted = new_attempts >= self.max_attempts
            next_state = UploadState.FAILED_EXHAUSTED if is_exhausted else UploadState.FAILED_RETRYABLE
            err_msg = f"Local file missing: {local_path}"
            backoff = min(self.backoff_max_sec, self.backoff_base_sec * (2 ** (new_attempts - 1)))
            next_at = (time.time() + backoff) if not is_exhausted else None
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._update_item_state(
                key,
                next_state,
                inc_attempt=True,
                error=err_msg,
                next_attempt_at=next_at,
                exhausted_at=now_iso if is_exhausted else None,
                exhausted_at_ts=time.time() if is_exhausted else None,
            )
            return False, err_msg

        expected_size = item["file_size_bytes"]
        remote = item["rclone_remote"]
        dest_path = item["rclone_dest_path"]

        # Step 1: Idempotency check: is it already verified remotely?
        verified, rem_sz, vmsg = self.verify_remote_file(filename, expected_size, remote, dest_path)
        if verified:
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._update_item_state(
                key,
                UploadState.UPLOADED,
                verified=True,
                uploaded_at=now_iso,
                next_attempt_at=None,
            )
            print(f"✅ Remote idempotent match: {key} already exists with {rem_sz} bytes.")
            return True, f"Idempotent match: {vmsg}"

        # Step 2: Mark UPLOADING and increment attempt count
        self._update_item_state(key, UploadState.UPLOADING, inc_attempt=True)
        new_attempts = current_attempts + 1

        def _record_failure(reason: str) -> Tuple[bool, str]:
            is_exhausted = (new_attempts >= self.max_attempts)
            next_state = UploadState.FAILED_EXHAUSTED if is_exhausted else UploadState.FAILED_RETRYABLE
            full_msg = (
                f"{reason} (attempt {new_attempts}/{self.max_attempts})"
                if not is_exhausted
                else f"{reason} (exhausted {new_attempts}/{self.max_attempts} attempts)"
            )
            backoff = min(self.backoff_max_sec, self.backoff_base_sec * (2 ** (new_attempts - 1)))
            next_at = (time.time() + backoff) if not is_exhausted else None
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._update_item_state(
                key,
                next_state,
                error=full_msg,
                next_attempt_at=next_at,
                exhausted_at=now_iso if is_exhausted else None,
                exhausted_at_ts=time.time() if is_exhausted else None,
            )
            print(f"⚠️ Upload {'exhausted' if is_exhausted else 'failed'}: {key} -> {full_msg}")
            return False, full_msg

        # Step 3: Execute single-file copyto command
        rclone_bin = shutil.which("rclone")
        if not rclone_bin:
            return _record_failure("rclone not found in PATH")

        if not dest_path:
            dst = f"{remote}{filename}"
            cmd = [rclone_bin, "copyto", str(local_path), dst]
        elif len(dest_path) > 20 and "/" not in dest_path:
            dst = f"{remote}{filename}"
            cmd = [rclone_bin, "copyto", str(local_path), dst, "--drive-root-folder-id", dest_path]
        else:
            dst = f"{remote.rstrip('/')}/{dest_path.lstrip('/')}/{filename}"
            cmd = [rclone_bin, "copyto", str(local_path), dst]

        print(f"🚀 Starting exact-file upload: {key} ({expected_size / (1024*1024):.1f} MB, attempt {new_attempts}/{self.max_attempts}) -> {dst}...")
        start_t = time.time()
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.upload_timeout_sec,
            )
            elapsed = time.time() - start_t

            if res.returncode != 0:
                err_msg = f"rclone copyto failed (rc={res.returncode}): {res.stderr.strip()}"
                return _record_failure(err_msg)

            # Step 4: Verify remote file after upload
            ver_ok, final_sz, ver_msg = self.verify_remote_file(filename, expected_size, remote, dest_path)
            if ver_ok:
                now_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                self._update_item_state(
                    key,
                    UploadState.UPLOADED,
                    verified=True,
                    uploaded_at=now_iso,
                    next_attempt_at=None,
                )
                print(f"✅ Upload & verified durable: {key} in {elapsed:.1f}s ({final_sz} bytes)")
                return True, f"Upload successful and remote verified in {elapsed:.1f}s"
            else:
                err_msg = f"Upload command succeeded but remote verification failed: {ver_msg}"
                return _record_failure(err_msg)

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_t
            err_msg = f"rclone copyto timed out after {self.upload_timeout_sec}s (elapsed {elapsed:.1f}s)"
            return _record_failure(err_msg)
        except Exception as e:
            err_msg = f"Unexpected upload exception: {e}"
            return _record_failure(err_msg)

    def _update_item_state(
        self,
        key_or_filename: str,
        state: str,
        inc_attempt: bool = False,
        error: Optional[str] = None,
        verified: Optional[bool] = None,
        uploaded_at: Optional[str] = None,
        next_attempt_at: Optional[float] = None,
        exhausted_at: Optional[str] = None,
        exhausted_at_ts: Optional[float] = None,
        reset_attempts: bool = False,
        recovery_cycle: Optional[int] = None,
    ):
        def _tx(data: Dict[str, Any]):
            items = data.setdefault("items", {})
            target_key = key_or_filename
            if target_key not in items:
                matches = [k for k, it in items.items() if it.get("filename") == key_or_filename]
                if len(matches) == 1:
                    target_key = matches[0]

            if target_key in items:
                item = items[target_key]
                item["state"] = state
                if reset_attempts:
                    item["attempt_count"] = 0
                elif inc_attempt:
                    item["attempt_count"] = item.get("attempt_count", 0) + 1
                    item["last_attempt_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                if error is not None:
                    item["last_error"] = str(error)
                if verified is not None:
                    item["remote_verified"] = verified
                if uploaded_at is not None:
                    item["uploaded_at"] = uploaded_at
                if next_attempt_at is not None:
                    item["next_attempt_at"] = next_attempt_at
                elif state in (UploadState.UPLOADED, UploadState.FAILED_EXHAUSTED):
                    item["next_attempt_at"] = None
                if exhausted_at is not None:
                    item["exhausted_at"] = exhausted_at
                if exhausted_at_ts is not None:
                    item["exhausted_at_ts"] = exhausted_at_ts
                if recovery_cycle is not None:
                    item["recovery_cycle"] = recovery_cycle

                if state == UploadState.FAILED_EXHAUSTED:
                    max_cycles = getattr(self, "max_recovery_cycles", DEFAULT_MAX_RECOVERY_CYCLES)
                    cooldown = getattr(self, "recovery_cooldown_sec", DEFAULT_RECOVERY_COOLDOWN_SEC)
                    if item.get("recovery_cycle", 0) < max_cycles:
                        ex_ts = item.get("exhausted_at_ts")
                        if ex_ts is None:
                            ex_ts = time.time()
                            item["exhausted_at_ts"] = ex_ts
                        item["earliest_recovery_ts"] = float(ex_ts) + cooldown
                        item["earliest_recovery_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(item["earliest_recovery_ts"]))
                    else:
                        item["earliest_recovery_ts"] = None
                        item["earliest_recovery_at"] = None
                elif state in (UploadState.PENDING, UploadState.UPLOADED):
                    item["earliest_recovery_ts"] = None
                    item["earliest_recovery_at"] = None

                recoverable = [it["earliest_recovery_ts"] for it in items.values() if it.get("earliest_recovery_ts") is not None]
                data["next_recovery_ts"] = min(recoverable) if recoverable else None

                return item, data
            return None, None

        self._locked_transaction(_tx)

    def reset_exhausted_items(
        self,
        cooldown_sec: Optional[float] = None,
        max_recovery_cycles: Optional[int] = None,
        now: Optional[float] = None,
    ) -> int:
        """
        Autonomous bounded recovery for FAILED_EXHAUSTED items after cooldown has elapsed.
        Increments recovery_cycle and resets attempt_count to 0, returning state to PENDING.
        If max_recovery_cycles has been reached or cooldown has not elapsed, item remains FAILED_EXHAUSTED.
        """
        rec_cooldown = cooldown_sec if cooldown_sec is not None else self.recovery_cooldown_sec
        max_cycles = max_recovery_cycles if max_recovery_cycles is not None else self.max_recovery_cycles
        current_time = now if now is not None else time.time()

        def _tx(data: Dict[str, Any]):
            recovered = 0
            items = data.setdefault("items", {})
            for k, item in items.items():
                if item.get("state") == UploadState.FAILED_EXHAUSTED:
                    is_eligible, _ = self.get_recovery_eligibility(
                        item,
                        now=current_time,
                        cooldown_sec=rec_cooldown,
                        max_recovery_cycles=max_cycles,
                    )
                    if is_eligible:
                        self._apply_recovery_cycle(item, max_recovery_cycles=max_cycles, reason="after cooldown")
                        recovered += 1
            if recovered > 0:
                recoverable = [it["earliest_recovery_ts"] for it in items.values() if it.get("earliest_recovery_ts") is not None]
                data["next_recovery_ts"] = min(recoverable) if recoverable else None
            return recovered, data if recovered > 0 else None

        recovered_count = self._locked_transaction(_tx)
        if recovered_count > 0:
            print(f"🔄 Recovered {recovered_count} exhausted upload(s) after cooldown.")
        return recovered_count

    def process_all_pending(self, max_items: Optional[int] = None) -> int:
        """Processes all unresolved items sequentially."""
        unresolved = self.get_unresolved_items()
        if max_items:
            unresolved = unresolved[:max_items]

        success_count = 0
        for item in unresolved:
            key = item.get("key", item.get("filename"))
            ok, msg = self.upload_file(key)
            if ok:
                success_count += 1
        return success_count

    def scan_and_register_untracked_clips(
        self,
        staging_dir: Path,
        camera_type: Optional[str] = None,
        rclone_remote: Optional[str] = None,
        rclone_dest_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scans staging directory for .mp4 clips. Uses camera targets authority to register
        any untracked clips with the correct camera identity and destination.
        """
        staging_dir = Path(staging_dir)
        if not staging_dir.exists():
            return []

        target = get_camera_target_for_path(staging_dir)
        cam = camera_type if camera_type is not None else target["camera_type"]
        remote = rclone_remote if rclone_remote is not None else target["rclone_remote"]
        dest = rclone_dest_path if rclone_dest_path is not None else target["rclone_dest_path"]

        registered = []
        for p in sorted(staging_dir.glob("*.mp4")):
            key = f"{cam}:{p.name}"
            item = self.get_item(key)
            if not item:
                try:
                    reg_item = self.register_file(
                        p,
                        camera_type=cam,
                        rclone_remote=remote,
                        rclone_dest_path=dest,
                    )
                    registered.append(reg_item)
                except Exception as e:
                    print(f"⚠️ Failed to auto-register {key}: {e}")
        return registered

    def get_earliest_recovery_ts(self, now: Optional[float] = None) -> Optional[float]:
        """
        Returns the earliest recovery eligibility timestamp across all unresolved items,
        or None if no items have remaining recovery cycles.
        """
        current_time = now if now is not None else time.time()
        earliest_ts = None
        for item in self.get_unresolved_items():
            if item.get("state") == UploadState.FAILED_EXHAUSTED:
                is_eligible, eligible_ts = self.get_recovery_eligibility(item, now=current_time)
                if is_eligible:
                    return current_time
                elif eligible_ts is not None:
                    earliest_ts = eligible_ts if earliest_ts is None else min(earliest_ts, eligible_ts)
            elif item.get("state") in (UploadState.PENDING, UploadState.FAILED_RETRYABLE, UploadState.UPLOADING):
                next_at = item.get("next_attempt_at")
                if next_at is not None:
                    earliest_ts = next_at if earliest_ts is None else min(earliest_ts, next_at)
                else:
                    return current_time
        return earliest_ts

    def schedule_wake(
        self,
        wake_ts: float,
        now: Optional[float] = None,
        service_name: str = DEFAULT_WAKE_SERVICE_NAME,
        timer_unit: str = DEFAULT_WAKE_TIMER_UNIT,
    ) -> Tuple[bool, str]:
        current_time = now if now is not None else time.time()
        wait_sec = max(1.0, wake_ts - current_time)
        return schedule_systemd_wake(wait_sec, service_name=service_name, timer_unit=timer_unit)

    def cancel_wake(self, timer_unit: str = DEFAULT_WAKE_TIMER_UNIT) -> Tuple[bool, str]:
        return cancel_systemd_wake(timer_unit=timer_unit)

    def run_until_empty(
        self,
        max_wait_sec: int = DEFAULT_MAX_WAIT_SEC,
        poll_interval_sec: float = 2.0,
        staging_dirs: Optional[List[Path]] = None,
        temp_dirs: Optional[List[Path]] = None,
        schedule_wake_on_cooldown: Optional[bool] = None,
        wake_service_name: str = DEFAULT_WAKE_SERVICE_NAME,
        wake_timer_unit: str = DEFAULT_WAKE_TIMER_UNIT,
    ) -> Tuple[bool, str]:
        """
        Processes queue items until empty, timeout, or all remaining items are FAILED_EXHAUSTED and unrecoverable.
        When successfully empty, invokes lifecycle_manager.complete_drain_if_ready().
        Fails closed on any lifecycle completion error or when all items are permanently exhausted.
        """
        print(f"🚀 Standalone uploader started (max_wait: {max_wait_sec}s)...")
        self.recover_pending()

        if staging_dirs is None:
            try:
                from scripts.lifecycle_manager import get_staging_dirs
                staging_dirs_to_check = get_staging_dirs()
            except Exception:
                staging_dirs_to_check = []
        else:
            staging_dirs_to_check = staging_dirs

        start_t = time.time()
        while time.time() - start_t < max_wait_sec:
            # Rescan staging directories to catch any newly arrived or untracked clips
            for sd in staging_dirs_to_check:
                self.scan_and_register_untracked_clips(sd)

            # Check for autonomous recovery of exhausted items
            self.reset_exhausted_items()

            unresolved = self.get_unresolved_items()
            if not unresolved:
                print("✅ All items in upload queue have been uploaded and remote verified.")
                try:
                    from scripts.lifecycle_manager import complete_drain_if_ready
                    ok, msg = complete_drain_if_ready(temp_dirs=temp_dirs, staging_dirs=staging_dirs_to_check)
                    if ok:
                        self.cancel_wake(wake_timer_unit)
                        print(f"✅ Drain finalized: {msg}")
                        return True, f"Uploads finished and drain completed: {msg}"
                    else:
                        print(f"⏳ Drain prerequisites not yet satisfied: {msg}")
                        time.sleep(poll_interval_sec)
                        continue
                except Exception as e:
                    err_msg = f"Lifecycle completion failed with exception: {e}"
                    print(f"🚨 {err_msg}")
                    return False, err_msg

            # Check if all remaining items are FAILED_EXHAUSTED
            all_exhausted = all(item.get("state") == UploadState.FAILED_EXHAUSTED for item in unresolved)
            if all_exhausted:
                # Determine whether any exhausted item has remaining recovery cycles
                recoverable_items = []
                earliest_eligible_ts = None
                now = time.time()
                for item in unresolved:
                    is_eligible, eligible_ts = self.get_recovery_eligibility(item, now=now)
                    if is_eligible:
                        recoverable_items.append(item)
                        earliest_eligible_ts = now if earliest_eligible_ts is None else min(earliest_eligible_ts, now)
                    elif eligible_ts is not None:
                        recoverable_items.append(item)
                        earliest_eligible_ts = eligible_ts if earliest_eligible_ts is None else min(earliest_eligible_ts, eligible_ts)

                if not recoverable_items:
                    # All items have exhausted ALL recovery cycles: fail closed immediately
                    self.cancel_wake(wake_timer_unit)
                    exhausted_names = [item.get("key", item.get("filename")) for item in unresolved]
                    err_msg = f"All remaining items ({len(unresolved)}) are permanently FAILED_EXHAUSTED (max cycles reached): {exhausted_names}"
                    print(f"🚨 Standalone uploader stopped: {err_msg}")
                    return False, err_msg

                # If cooldown has already elapsed, loop immediately so reset_exhausted_items recovers it
                if earliest_eligible_ts is not None and time.time() >= earliest_eligible_ts:
                    continue

                # Items are recoverable in the future.
                use_wake = schedule_wake_on_cooldown
                if use_wake is None:
                    sctl = shutil.which("systemctl")
                    srun = shutil.which("systemd-run")
                    use_wake = bool(sctl and "systemctl" in sctl and srun and "systemd-run" in srun)

                wait_sec = max(0.0, (earliest_eligible_ts - time.time())) if earliest_eligible_ts is not None else 0.0

                if use_wake:
                    self.schedule_wake(earliest_eligible_ts, service_name=wake_service_name, timer_unit=wake_timer_unit)
                    iso_ts = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(earliest_eligible_ts)) if earliest_eligible_ts else ""
                    msg = f"All {len(unresolved)} remaining item(s) are in recovery cooldown. Scheduled wake at {iso_ts} ({wait_sec:.1f}s). Exiting bounded invocation."
                    print(f"⏳ {msg}")
                    return False, msg

                # Fallback: sleep in-process bounded by remaining budget (for mock/non-systemd environments)
                remaining_budget = max_wait_sec - (time.time() - start_t)
                if remaining_budget <= 0:
                    break

                sleep_duration = min(wait_sec, remaining_budget)
                sleep_target = min(sleep_duration + 0.1, remaining_budget)
                if sleep_target > 0:
                    print(f"⏳ All {len(unresolved)} remaining item(s) are in recovery cooldown. Sleeping {sleep_target:.1f}s until earliest recovery eligibility...")
                    time.sleep(sleep_target)
                continue

            # Process ready items
            now = time.time()
            for item in unresolved:
                state = item.get("state")
                if state == UploadState.FAILED_EXHAUSTED:
                    continue
                if state == UploadState.FAILED_RETRYABLE:
                    next_at = item.get("next_attempt_at")
                    if next_at and now < next_at:
                        continue

                key = item.get("key", item.get("filename"))
                self.upload_file(key)

            time.sleep(poll_interval_sec)

        unres = self.get_unresolved_items()
        earliest_ts = self.get_earliest_recovery_ts()
        use_wake = schedule_wake_on_cooldown
        if use_wake is None:
            sctl = shutil.which("systemctl")
            srun = shutil.which("systemd-run")
            use_wake = bool(sctl and "systemctl" in sctl and srun and "systemd-run" in srun)

        if earliest_ts is not None and use_wake:
            now = time.time()
            wait_sec = max(0.0, earliest_ts - now)
            self.schedule_wake(earliest_ts, now=now, service_name=wake_service_name, timer_unit=wake_timer_unit)
            err_msg = f"Standalone uploader timed out after {max_wait_sec}s with {len(unres)} unresolved item(s). Scheduled future recovery wake in {wait_sec:.1f}s."
        else:
            err_msg = f"Standalone uploader timed out after {max_wait_sec}s with {len(unres)} unresolved item(s)."

        print(f"🚨 {err_msg}")
        return False, err_msg


class UploadQueueWorker:
    """
    Background worker that monitors the upload queue, ensuring bounded single-worker concurrency
    with automatic retry and exponential backoff.
    """

    def __init__(self, upload_queue: UploadQueue, poll_interval_sec: float = 5.0):
        self.queue = upload_queue
        self.poll_interval = poll_interval_sec
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def start(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._wake_event.clear()
        self._worker_thread = threading.Thread(target=self._run_loop, name="FairFeederUploadWorker", daemon=True)
        self._worker_thread.start()
        print("🧵 UploadQueueWorker background thread started.")

    def notify(self):
        """Signals the worker thread that new items are ready for upload."""
        self._wake_event.set()

    def stop(self, timeout: float = 10.0):
        self._stop_event.set()
        self._wake_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        print("🛑 UploadQueueWorker stopped.")

    def _run_loop(self):
        while not self._stop_event.is_set():
            # Check for unresolved items
            unresolved = self.queue.get_unresolved_items()
            now = time.time()
            if unresolved:
                for item in unresolved:
                    if self._stop_event.is_set():
                        break

                    state = item.get("state")
                    if state == UploadState.FAILED_EXHAUSTED:
                        continue

                    if state == UploadState.FAILED_RETRYABLE:
                        next_at = item.get("next_attempt_at")
                        if next_at and now < next_at:
                            continue

                    key = item.get("key", item.get("filename"))
                    try:
                        self.queue.upload_file(key)
                    except Exception as e:
                        print(f"⚠️ UploadQueueWorker error processing {key}: {e}")

            # Wait for wake event or timeout
            self._wake_event.wait(timeout=self.poll_interval)
            self._wake_event.clear()


def main():
    import argparse
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--ledger-path", default=None, help="Explicit path to ledger JSON file")

    parser = argparse.ArgumentParser(description="Fair Feeder Durable Upload Queue CLI", parents=[parent_parser])
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Show upload ledger status", parents=[parent_parser])
    subparsers.add_parser("process", help="Process all pending uploads", parents=[parent_parser])
    subparsers.add_parser("recover", help="Recover in-flight uploads", parents=[parent_parser])

    run_empty_p = subparsers.add_parser("run-until-empty", help="Run uploader until queue is empty and trigger drain completion", parents=[parent_parser])
    run_empty_p.add_argument("--max-wait-sec", type=int, default=DEFAULT_MAX_WAIT_SEC, help="Max seconds to wait")
    run_empty_p.add_argument("--poll-interval", type=float, default=2.0, help="Poll interval in seconds")
    run_empty_p.add_argument("--no-schedule-wake", action="store_true", help="Disable systemd wake scheduling on cooldown")
    run_empty_p.add_argument("--wake-service", default=DEFAULT_WAKE_SERVICE_NAME, help="Systemd service unit to wake")
    run_empty_p.add_argument("--wake-timer", default=DEFAULT_WAKE_TIMER_UNIT, help="Systemd transient timer unit name")

    reg_p = subparsers.add_parser("register", help="Register a local file for upload", parents=[parent_parser])
    reg_p.add_argument("filepath", help="Path to video file")
    reg_p.add_argument("--camera", default="rtsp", help="Camera type (rtsp/usb)")
    reg_p.add_argument("--remote", default="gdrive-randomdice:", help="rclone remote")
    reg_p.add_argument("--dest-path", default="", help="rclone destination path or folder ID")

    args = parser.parse_args()
    ledger_p = getattr(args, "ledger_path", None)
    queue = UploadQueue(ledger_path=ledger_p)

    if args.command == "status":
        items = queue.get_all_items()
        unresolved = queue.get_unresolved_items()
        print(f"Upload Ledger ({queue.ledger_path}):")
        print(f"  Total registered items: {len(items)}")
        print(f"  Unresolved items: {len(unresolved)}")
        for name, item in items.items():
            print(f"    - {name}: state={item.get('state')}, size={item.get('file_size_bytes')}, attempts={item.get('attempt_count')}, verified={item.get('remote_verified')}")
    elif args.command == "process":
        count = queue.process_all_pending()
        print(f"Processed pending uploads: {count} succeeded.")
    elif args.command == "recover":
        rec = queue.recover_pending()
        print(f"Recovered {rec} item(s).")
    elif args.command == "run-until-empty":
        schedule_wake = False if args.no_schedule_wake else None
        ok, msg = queue.run_until_empty(
            max_wait_sec=args.max_wait_sec,
            poll_interval_sec=args.poll_interval,
            schedule_wake_on_cooldown=schedule_wake,
            wake_service_name=args.wake_service,
            wake_timer_unit=args.wake_timer,
        )
        print(msg)
        sys.exit(0 if ok else 1)
    elif args.command == "register":
        p = Path(args.filepath)
        item = queue.register_file(p, camera_type=args.camera, rclone_remote=args.remote, rclone_dest_path=args.dest_path)
        print(json.dumps(item, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
