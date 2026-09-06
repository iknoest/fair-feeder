"""
scripts/weight_store.py - Canonical Weight Store for Fair Feeder

Provides a single authoritative interface for reading, writing, merging, and syncing
Dan and Sanbo weight records.
Shared by:
- Telegram /weight (log, edit, history, chart)
- Telegram /profile (latest weight, delta, trend, history)
- Motion recorder weight logging
- Morning report / Drive sync
"""

import os
import sys
import csv
import shutil
import logging
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

try:
    import fcntl
except ImportError:
    fcntl = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("WeightStore")

VALID_CATS = {"dan", "sanbo"}
DEFAULT_REMOTE = os.environ.get("RCLONE_REMOTE", "gdrive-randomdice:")


class WeightConflictError(Exception):
    """Raised when conflicting weights exist for the same (cat, date) pair."""
    pass


class WeightValidationError(Exception):
    """Raised when weight record data is invalid."""
    pass


class WeightCorruptError(Exception):
    """Raised when canonical weight file is unreadable, corrupt, or contains malformed data."""
    pass


def get_canonical_weight_path() -> Path:
    """
    Resolves the single canonical path for weight_log.csv.
    Priority:
    1. WEIGHT_LOG_PATH environment variable (for testing/overrides)
    2. /home/pi5/Pictures/gdrive-randomdice-sync/weight_log.csv (Pi production staging)
    3. REPO_ROOT / weight_log.csv (local development / fallback)
    """
    env_override = os.environ.get("WEIGHT_LOG_PATH")
    if env_override:
        return Path(env_override)

    pi_staging_dir = Path("/home/pi5/Pictures/gdrive-randomdice-sync")
    if pi_staging_dir.exists():
        return pi_staging_dir / "weight_log.csv"

    return REPO_ROOT / "weight_log.csv"


def get_secondary_weight_paths() -> List[Path]:
    """Returns secondary paths that should stay in sync with the canonical file if they exist."""
    if os.environ.get("WEIGHT_LOG_PATH"):
        return []
    canonical = get_canonical_weight_path().resolve()
    candidates = [
        REPO_ROOT / "weight_log.csv",
        Path("/home/pi5/Pictures/usb-camera-sync/weight_log.csv"),
    ]
    secondary = []
    for p in candidates:
        if p.parent.exists() and p.resolve() != canonical:
            secondary.append(p)
    return secondary


def validate_weight_record(record: Dict[str, Any]) -> Dict[str, str]:
    """Validates and normalizes a weight record."""
    if not isinstance(record, dict):
        raise WeightValidationError(f"Record must be a dict, got {type(record)}")

    raw_date = str(record.get("date", "")).strip()
    raw_cat = str(record.get("cat", "")).strip().lower()
    raw_weight = str(record.get("weight_kg", "")).strip()

    # Validate date
    try:
        dt = datetime.strptime(raw_date, "%Y-%m-%d")
        normalized_date = dt.strftime("%Y-%m-%d")
    except ValueError:
        raise WeightValidationError(f"Invalid date format '{raw_date}'. Expected YYYY-MM-DD.")

    # Validate cat
    if raw_cat not in VALID_CATS:
        raise WeightValidationError(f"Invalid cat '{raw_cat}'. Expected one of {sorted(VALID_CATS)}.")

    # Validate weight
    try:
        w = float(raw_weight)
        if w <= 0 or w > 25.0:
            raise WeightValidationError(f"Weight {w} kg out of realistic range (0 - 25 kg).")
        normalized_weight = f"{w:.2f}"
    except (ValueError, TypeError):
        raise WeightValidationError(f"Invalid weight_kg '{raw_weight}'. Expected positive number.")

    return {
        "date": normalized_date,
        "cat": raw_cat,
        "weight_kg": normalized_weight,
    }


def merge_weight_rows(*datasets: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Merges multiple weight datasets with deduplication and strict conflict guarding.
    Key: (cat, date)
    - If same (cat, date) and same weight: deduplicated safely.
    - If same (cat, date) and different weight: raises WeightConflictError.
    """
    merged_map: Dict[Tuple[str, str], Dict[str, str]] = {}

    for dataset in datasets:
        for row in dataset:
            clean_row = validate_weight_record(row)
            key = (clean_row["cat"], clean_row["date"])

            if key in merged_map:
                existing = merged_map[key]
                if existing["weight_kg"] != clean_row["weight_kg"]:
                    raise WeightConflictError(
                        f"Weight conflict for {key[0].capitalize()} on {key[1]}: "
                        f"existing={existing['weight_kg']} kg vs new={clean_row['weight_kg']} kg"
                    )
                # Exact duplicate, safely ignore
            else:
                merged_map[key] = clean_row

    # Sort deterministically by date ascending, then cat (dan before sanbo)
    sorted_rows = sorted(
        merged_map.values(),
        key=lambda r: (r["date"], 0 if r["cat"] == "dan" else 1)
    )
    return sorted_rows


def load_weights(path: Optional[Path] = None) -> List[Dict[str, str]]:
    """
    Reads all weight rows from the canonical or specified file.
    Fails closed: raises WeightCorruptError if the file exists but is unreadable,
    has missing/invalid headers, or contains malformed rows.
    """
    if path is not None:
        target = path
        if not target.exists():
            return []
    else:
        target = get_canonical_weight_path()
        if not target.exists():
            fallback = REPO_ROOT / "weight_log.csv"
            if target != fallback and fallback.exists():
                target = fallback
            else:
                return []

    if target.stat().st_size == 0:
        return []

    try:
        with open(target, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {"date", "cat", "weight_kg"}.issubset(set(reader.fieldnames)):
                raise WeightCorruptError(
                    f"Weight file {target} missing required header fields: date, cat, weight_kg (found: {reader.fieldnames})"
                )
            rows = []
            for line_idx, r in enumerate(reader, start=2):
                try:
                    rows.append(validate_weight_record(r))
                except WeightValidationError as e:
                    raise WeightCorruptError(
                        f"Malformed weight row at line {line_idx} in {target}: {r}. Error: {e}. Preserving file and failing closed."
                    )
            return sorted(rows, key=lambda r: (r["date"], 0 if r["cat"] == "dan" else 1))
    except WeightCorruptError:
        raise
    except Exception as e:
        raise WeightCorruptError(f"Failed to read weight file {target}: {e}")


def save_weights(
    rows: List[Dict[str, Any]],
    path: Optional[Path] = None,
    sync_drive: bool = True,
    rclone_remote: str = DEFAULT_REMOTE,
) -> bool:
    """
    Atomically writes rows to the canonical weight file with file locking,
    replicates to secondary paths, and executes verified sync to Google Drive.
    Fails closed: refuses to overwrite an existing non-empty file if it is corrupt,
    and refuses to overwrite an existing non-empty file with empty rows.
    """
    target = path or get_canonical_weight_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    # Fail closed: if target exists and has data, verify it is healthy before overwriting
    if target.exists() and target.stat().st_size > 0:
        # load_weights will raise WeightCorruptError if existing target is unreadable or malformed
        load_weights(target)
        if not rows:
            raise WeightCorruptError(f"Refusing to overwrite existing non-empty weight file {target} with empty data.")

    # Validate and deduplicate all rows before writing
    clean_rows = merge_weight_rows(rows)

    lock_path = target.with_suffix(".lock")
    lock_file = open(lock_path, "w")
    try:
        if fcntl:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        # Atomic write to temporary file on same filesystem
        with tempfile.NamedTemporaryFile("w", dir=str(target.parent), delete=False, newline="", encoding="utf-8") as tf:
            writer = csv.DictWriter(tf, fieldnames=["date", "cat", "weight_kg"])
            writer.writeheader()
            writer.writerows(clean_rows)
            temp_path = Path(tf.name)

        # Atomic rename
        os.replace(temp_path, target)

        # Mirror to secondary local paths if canonical
        if path is None or path == get_canonical_weight_path():
            for sec in get_secondary_weight_paths():
                try:
                    sec.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, sec)
                except Exception as e:
                    log.warning(f"Failed to replicate weight file to secondary path {sec}: {e}")

    finally:
        if fcntl:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        lock_file.close()

    if sync_drive:
        sync_ok = sync_weight_to_drive(target, remote=rclone_remote)
        if not sync_ok:
            log.warning("Drive sync did not complete successfully.")
            return False
    return True


def sync_weight_to_drive(
    local_path: Optional[Path] = None,
    remote: str = DEFAULT_REMOTE,
    timeout_sec: int = 15,
) -> bool:
    """
    Durable exact-file sync to Google Drive using rclone copyto with remote verification.
    Ensures local writes are durably acknowledged remotely without silent failures.
    """
    target = local_path or get_canonical_weight_path()
    if not target.exists():
        log.error(f"Cannot sync non-existent weight file {target}")
        return False

    remote_dest = remote.rstrip("/") + "/weight_log.csv"
    try:
        # 1. Exact-file copy
        copy_res = subprocess.run(
            ["rclone", "copyto", str(target), remote_dest],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if copy_res.returncode != 0:
            log.error(f"rclone copyto failed (code {copy_res.returncode}): {copy_res.stderr.strip()}")
            return False

        # 2. Remote verification
        check_res = subprocess.run(
            ["rclone", "size", remote_dest, "--json"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if check_res.returncode != 0:
            log.error(f"rclone verification check failed: {check_res.stderr.strip()}")
            return False

        import json
        size_data = json.loads(check_res.stdout)
        remote_bytes = size_data.get("bytes", 0)
        local_bytes = target.stat().st_size

        if remote_bytes != local_bytes:
            log.error(f"Remote size mismatch: remote={remote_bytes} bytes vs local={local_bytes} bytes")
            return False

        log.info(f"Verified weight_log.csv durable sync to {remote_dest} ({remote_bytes} bytes)")
        return True

    except subprocess.TimeoutExpired:
        log.error(f"rclone sync timed out after {timeout_sec}s")
        return False
    except FileNotFoundError:
        log.warning("rclone binary not found in PATH; skipping Drive sync")
        return False
    except Exception as e:
        log.error(f"Unexpected error during Drive weight sync: {e}")
        return False


def get_cat_weight_summary(cat: str, rows: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Computes latest weight, previous weight, delta, and trend for a cat.
    """
    cat_lower = cat.strip().lower()
    if rows is None:
        rows = load_weights()

    cat_rows = [r for r in rows if r["cat"] == cat_lower]
    cat_rows.sort(key=lambda r: r["date"])

    if not cat_rows:
        return {
            "cat": cat_lower,
            "latest_weight": None,
            "latest_date": None,
            "previous_weight": None,
            "previous_date": None,
            "change": None,
            "trend": "no_data",
            "count": 0,
            "history": [],
        }

    latest = float(cat_rows[-1]["weight_kg"])
    latest_date = cat_rows[-1]["date"]

    previous = None
    previous_date = None
    change = None
    trend = "single_point"

    if len(cat_rows) >= 2:
        previous = float(cat_rows[-2]["weight_kg"])
        previous_date = cat_rows[-2]["date"]
        change = round(latest - previous, 2)
        if change > 0.05:
            trend = "gaining"
        elif change < -0.05:
            trend = "losing"
        else:
            trend = "stable"

    return {
        "cat": cat_lower,
        "latest_weight": latest,
        "latest_date": latest_date,
        "previous_weight": previous,
        "previous_date": previous_date,
        "change": change,
        "trend": trend,
        "count": len(cat_rows),
        "history": cat_rows,
    }
