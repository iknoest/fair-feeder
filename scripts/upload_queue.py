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
DEFAULT_LEDGER_PATH = repo_root / "upload_ledger.json"
DEFAULT_UPLOAD_TIMEOUT_SEC = 600
DEFAULT_VERIFY_TIMEOUT_SEC = 30
DEFAULT_MAX_ATTEMPTS = 5


class UploadState:
    PENDING = "PENDING"
    UPLOADING = "UPLOADING"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    UPLOADED = "UPLOADED"


class UploadQueue:
    """Manages persistent per-file upload state and execution."""

    def __init__(
        self,
        ledger_path: Optional[Path] = None,
        upload_timeout_sec: int = DEFAULT_UPLOAD_TIMEOUT_SEC,
        verify_timeout_sec: int = DEFAULT_VERIFY_TIMEOUT_SEC,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ):
        if ledger_path is None:
            env_path = os.environ.get("UPLOAD_LEDGER_PATH")
            self.ledger_path = Path(env_path) if env_path else DEFAULT_LEDGER_PATH
        else:
            self.ledger_path = Path(ledger_path)

        self.upload_timeout_sec = upload_timeout_sec
        self.verify_timeout_sec = verify_timeout_sec
        self.max_attempts = max_attempts
        self._thread_lock = threading.Lock()
        self._init_ledger()

    def _get_lock_path(self) -> Path:
        return self.ledger_path.with_suffix(".lock")

    def _init_ledger(self):
        with self._thread_lock:
            if not self.ledger_path.exists():
                self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
                init_data = {
                    "version": 1,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "items": {},
                }
                self._write_ledger_unlocked(init_data)

    def _read_ledger_unlocked(self) -> Dict[str, Any]:
        if not self.ledger_path.exists():
            return {"version": 1, "updated_at": None, "items": {}}
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"version": 1, "updated_at": None, "items": {}}

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

    def register_file(
        self,
        filepath: Path,
        camera_type: str = "rtsp",
        rclone_remote: str = "gdrive-randomdice:",
        rclone_dest_path: str = "",
    ) -> Dict[str, Any]:
        """
        Durably registers a finalized local file into the ledger as PENDING.
        Must be called BEFORE upload begins.
        """
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"Cannot register non-existent file for upload: {filepath}")

        file_size = filepath.stat().st_size
        filename = filepath.name

        def _tx(data: Dict[str, Any]):
            items = data.setdefault("items", {})
            existing = items.get(filename)
            if existing and existing.get("state") == UploadState.UPLOADED and existing.get("remote_verified"):
                return existing, None

            now_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            item = {
                "filename": filename,
                "filepath": str(filepath),
                "camera_type": camera_type,
                "rclone_remote": rclone_remote,
                "rclone_dest_path": rclone_dest_path,
                "file_size_bytes": file_size,
                "state": UploadState.PENDING,
                "attempt_count": existing.get("attempt_count", 0) if existing else 0,
                "created_at": existing.get("created_at", now_iso) if existing else now_iso,
                "last_attempt_at": existing.get("last_attempt_at") if existing else None,
                "last_error": None,
                "remote_verified": False,
                "uploaded_at": None,
            }
            items[filename] = item
            return item, data

        registered_item = self._locked_transaction(_tx)
        print(f"📋 Registered for durable upload: {filename} ({file_size / (1024*1024):.1f} MB, state={UploadState.PENDING})")
        return registered_item

    def get_item(self, filename: str) -> Optional[Dict[str, Any]]:
        def _tx(data: Dict[str, Any]):
            return data.get("items", {}).get(filename), None
        return self._locked_transaction(_tx)

    def get_all_items(self) -> Dict[str, Dict[str, Any]]:
        def _tx(data: Dict[str, Any]):
            return dict(data.get("items", {})), None
        return self._locked_transaction(_tx)

    def get_unresolved_items(self) -> List[Dict[str, Any]]:
        """Returns all items in PENDING, UPLOADING, or FAILED_RETRYABLE states."""
        def _tx(data: Dict[str, Any]):
            unresolved = [
                item for item in data.get("items", {}).values()
                if item.get("state") in (UploadState.PENDING, UploadState.UPLOADING, UploadState.FAILED_RETRYABLE)
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
        names = [u["filename"] for u in unresolved[:3]]
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
            for name, item in items.items():
                if item.get("state") == UploadState.UPLOADING:
                    item["state"] = UploadState.PENDING
                    item["last_error"] = "Process restarted while upload in-flight"
                    recovered_count += 1

                # Check file existence if pending or retryable
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

    def upload_file(self, filename: str) -> Tuple[bool, str]:
        """
        Uploads a single exact file with full lifecycle:
        1. Pre-check idempotency (already on remote).
        2. Set state = UPLOADING.
        3. Run exact rclone copyto.
        4. Verify remote file existence and size.
        5. Set state = UPLOADED on success, or FAILED_RETRYABLE on failure.
        """
        item = self.get_item(filename)
        if not item:
            return False, f"File not registered in upload ledger: {filename}"

        if item.get("state") == UploadState.UPLOADED and item.get("remote_verified"):
            return True, f"Already uploaded and verified: {filename}"

        local_path = Path(item["filepath"])
        if not local_path.exists():
            err_msg = f"Local file missing: {local_path}"
            self._update_item_state(filename, UploadState.FAILED_RETRYABLE, error=err_msg)
            return False, err_msg

        expected_size = item["file_size_bytes"]
        remote = item["rclone_remote"]
        dest_path = item["rclone_dest_path"]

        # Step 1: Idempotency check: is it already verified remotely?
        verified, rem_sz, vmsg = self.verify_remote_file(filename, expected_size, remote, dest_path)
        if verified:
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._update_item_state(filename, UploadState.UPLOADED, verified=True, uploaded_at=now_iso)
            print(f"✅ Remote idempotent match: {filename} already exists with {rem_sz} bytes.")
            return True, f"Idempotent match: {vmsg}"

        # Step 2: Mark UPLOADING
        self._update_item_state(filename, UploadState.UPLOADING, inc_attempt=True)

        # Step 3: Execute single-file copyto command
        rclone_bin = shutil.which("rclone")
        if not rclone_bin:
            err_msg = "rclone not found in PATH"
            self._update_item_state(filename, UploadState.FAILED_RETRYABLE, error=err_msg)
            return False, err_msg

        if not dest_path:
            dst = f"{remote}{filename}"
            cmd = [rclone_bin, "copyto", str(local_path), dst]
        elif len(dest_path) > 20 and "/" not in dest_path:
            dst = f"{remote}{filename}"
            cmd = [rclone_bin, "copyto", str(local_path), dst, "--drive-root-folder-id", dest_path]
        else:
            dst = f"{remote.rstrip('/')}/{dest_path.lstrip('/')}/{filename}"
            cmd = [rclone_bin, "copyto", str(local_path), dst]

        print(f"🚀 Starting exact-file upload: {filename} ({expected_size / (1024*1024):.1f} MB) -> {dst}...")
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
                self._update_item_state(filename, UploadState.FAILED_RETRYABLE, error=err_msg)
                print(f"⚠️ Upload failed for {filename} after {elapsed:.1f}s: {err_msg}")
                return False, err_msg

            # Step 4: Verify remote file after upload
            ver_ok, final_sz, ver_msg = self.verify_remote_file(filename, expected_size, remote, dest_path)
            if ver_ok:
                now_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                self._update_item_state(filename, UploadState.UPLOADED, verified=True, uploaded_at=now_iso)
                print(f"✅ Upload & verified durable: {filename} in {elapsed:.1f}s ({final_sz} bytes)")
                return True, f"Upload successful and remote verified in {elapsed:.1f}s"
            else:
                err_msg = f"Upload command succeeded but remote verification failed: {ver_msg}"
                self._update_item_state(filename, UploadState.FAILED_RETRYABLE, error=err_msg)
                print(f"⚠️ Verification failed for {filename}: {err_msg}")
                return False, err_msg

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_t
            err_msg = f"rclone copyto timed out after {self.upload_timeout_sec}s (elapsed {elapsed:.1f}s)"
            self._update_item_state(filename, UploadState.FAILED_RETRYABLE, error=err_msg)
            print(f"⚠️ Upload timed out for {filename}: {err_msg}")
            return False, err_msg
        except Exception as e:
            err_msg = f"Unexpected upload exception: {e}"
            self._update_item_state(filename, UploadState.FAILED_RETRYABLE, error=err_msg)
            print(f"⚠️ Upload exception for {filename}: {err_msg}")
            return False, err_msg

    def _update_item_state(
        self,
        filename: str,
        state: str,
        inc_attempt: bool = False,
        error: Optional[str] = None,
        verified: Optional[bool] = None,
        uploaded_at: Optional[str] = None,
    ):
        def _tx(data: Dict[str, Any]):
            items = data.setdefault("items", {})
            if filename in items:
                item = items[filename]
                item["state"] = state
                if inc_attempt:
                    item["attempt_count"] = item.get("attempt_count", 0) + 1
                    item["last_attempt_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                if error is not None:
                    item["last_error"] = str(error)
                if verified is not None:
                    item["remote_verified"] = verified
                if uploaded_at is not None:
                    item["uploaded_at"] = uploaded_at
                return item, data
            return None, None

        self._locked_transaction(_tx)

    def process_all_pending(self, max_items: Optional[int] = None) -> int:
        """Processes all unresolved items sequentially."""
        unresolved = self.get_unresolved_items()
        if max_items:
            unresolved = unresolved[:max_items]

        success_count = 0
        for item in unresolved:
            fname = item["filename"]
            ok, msg = self.upload_file(fname)
            if ok:
                success_count += 1
        return success_count


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
            if unresolved:
                for item in unresolved:
                    if self._stop_event.is_set():
                        break

                    filename = item["filename"]
                    attempts = item.get("attempt_count", 0)

                    # Bounded exponential backoff if previously failed
                    if item.get("state") == UploadState.FAILED_RETRYABLE and attempts > 0:
                        backoff = min(60.0, 5.0 * (2 ** min(attempts - 1, 4)))
                        time.sleep(min(backoff, 5.0))

                    try:
                        self.queue.upload_file(filename)
                    except Exception as e:
                        print(f"⚠️ UploadQueueWorker error processing {filename}: {e}")

            # Wait for wake event or timeout
            self._wake_event.wait(timeout=self.poll_interval)
            self._wake_event.clear()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fair Feeder Durable Upload Queue CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Show upload ledger status")
    subparsers.add_parser("process", help="Process all pending uploads")
    subparsers.add_parser("recover", help="Recover in-flight uploads")

    reg_p = subparsers.add_parser("register", help="Register a local file for upload")
    reg_p.add_argument("filepath", help="Path to video file")
    reg_p.add_argument("--camera", default="rtsp", help="Camera type (rtsp/usb)")
    reg_p.add_argument("--remote", default="gdrive-randomdice:", help="rclone remote")
    reg_p.add_argument("--dest-path", default="", help="rclone destination path or folder ID")

    args = parser.parse_args()
    queue = UploadQueue()

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
    elif args.command == "register":
        p = Path(args.filepath)
        item = queue.register_file(p, camera_type=args.camera, rclone_remote=args.remote, rclone_dest_path=args.dest_path)
        print(json.dumps(item, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
