#!/usr/bin/env python3
"""
Fair Feeder Shared-Host Idle Preflight Checker

Provides a clean, host-observable contract for other host workloads (such as Jobsearcher)
to verify that Fair Feeder has finished its morning feeding duties, drained local recordings,
unloaded heavy recorder processes, and released host RAM before starting heavy tasks.
"""

import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.lifecycle_manager import read_state, is_service_active, get_active_workers_count, get_mem_available_mb, LifecycleState

MIN_MEM_AVAILABLE_MB = 1200  # Minimum MB free for heavy host workloads

def check_host_idle() -> dict:
    state_data = read_state()
    current_state = state_data.get("state", LifecycleState.UNKNOWN)
    drain_complete = state_data.get("local_drain_complete", False)

    cat_monitor_active = is_service_active("cat-monitor")
    usb_monitor_active = is_service_active("usb-monitor")
    services_active = cat_monitor_active or usb_monitor_active

    active_workers = get_active_workers_count()
    mem_available = get_mem_available_mb()
    mem_ok = mem_available >= MIN_MEM_AVAILABLE_MB

    is_idle = (
        current_state == LifecycleState.DAYTIME_IDLE
        and drain_complete is True
        and services_active is False
        and active_workers == 0
    )

    ready_for_workload = is_idle and mem_ok

    return {
        "fair_feeder_idle": is_idle,
        "state": current_state,
        "local_drain_complete": drain_complete,
        "services_active": services_active,
        "cat_monitor_active": cat_monitor_active,
        "usb_monitor_active": usb_monitor_active,
        "active_workers": active_workers,
        "mem_available_mb": mem_available,
        "mem_threshold_met": mem_ok,
        "ready_for_heavy_workload": ready_for_workload,
        "last_event": state_data.get("last_event", "")
    }

def main():
    res = check_host_idle()
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["ready_for_heavy_workload"] else 1)

if __name__ == "__main__":
    main()
