#!/usr/bin/env python3
"""
scripts/validate_tapo_artifact.py

Validates the TAPO evidence artifact bundle before upload in GitHub Actions.
Ensures fail-closed behavior: if TAPO feeding was detected/analyzed upstream,
the artifact bundle MUST contain:
1. tapo_summary_{TARGET_DATE}.json (non-empty, valid JSON)
2. tapo_timeline_{TARGET_DATE}.json (valid JSON)
3. At least one source feeding video clip matching motion_{TARGET_DATE}_*.mp4 (> 0 bytes)

If no feeding footage genuinely existed for the date, validation passes
allowing zero videos only when timeline records zero feeding phases.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def validate_tapo_evidence_bundle(
    evidence_dir: Path,
    target_date: str,
    strict_feeding_required: bool = False
) -> Dict[str, Any]:
    """
    Validates evidence bundle in evidence_dir for target_date.
    Returns validation result dict. Raises ValueError on contract violation.
    """
    clean_date = str(target_date).replace("-", "").strip()
    evidence_dir = Path(evidence_dir)

    if not evidence_dir.exists() or not evidence_dir.is_dir():
        raise ValueError(f"Evidence directory does not exist or is not a directory: {evidence_dir}")

    summary_file = evidence_dir / f"tapo_summary_{clean_date}.json"
    timeline_file = evidence_dir / f"tapo_timeline_{clean_date}.json"

    # Collect valid source feeding clips (exclude combined and annotated)
    source_clips: List[Path] = []
    for f in sorted(evidence_dir.glob(f"*{clean_date}*.mp4")):
        if "combined" not in f.name and "annotated" not in f.name and f.stat().st_size > 0:
            source_clips.append(f)

    # 1. Timeline inspection
    timeline_data: Dict[str, Any] = {}
    has_timeline = False
    feeding_phases: List[Dict[str, Any]] = []
    if timeline_file.exists():
        try:
            timeline_data = json.loads(timeline_file.read_text(encoding="utf-8"))
            has_timeline = True
            feeding_phases = timeline_data.get("feeding_phases", [])
        except Exception as e:
            raise ValueError(f"Corrupted or invalid timeline JSON in {timeline_file}: {e}")

    # 2. Summary inspection
    summary_data: Dict[str, Any] = {}
    has_summary = False
    if summary_file.exists():
        try:
            summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
            if summary_data:
                has_summary = True
        except Exception as e:
            raise ValueError(f"Corrupted or invalid summary JSON in {summary_file}: {e}")

    # Determine if feeding evidence was detected upstream
    dan_kibble = summary_data.get("dan_kibble") or summary_data.get("dan_kibble_eaten") or 0
    sanbo_kibble = summary_data.get("sanbo_kibble") or summary_data.get("sanbo_kibble_eaten") or 0
    start_kibble = summary_data.get("start_kibble")
    dan_bowl_sec = summary_data.get("dan_bowl_seconds") or 0
    sanbo_bowl_sec = summary_data.get("sanbo_bowl_seconds") or 0

    feeding_detected = bool(
        len(feeding_phases) > 0 or
        dan_kibble > 0 or
        sanbo_kibble > 0 or
        dan_bowl_sec > 0 or
        sanbo_bowl_sec > 0 or
        (start_kibble is not None and start_kibble > 0)
    )

    if strict_feeding_required and not feeding_detected:
        raise ValueError(f"Strict feeding validation required, but no feeding evidence detected for {clean_date}")

    # Contract Enforcement
    if feeding_detected:
        errors = []
        if not has_summary:
            errors.append(f"Missing required tapo_summary_{clean_date}.json")
        if not has_timeline:
            errors.append(f"Missing required tapo_timeline_{clean_date}.json")
        if not source_clips:
            errors.append(
                f"Missing TAPO source feeding video(s). Feeding was detected upstream "
                f"({len(feeding_phases)} phases, {dan_kibble}g Dan kibble), but 0 source clips "
                f"matching motion_{clean_date}_*.mp4 exist in {evidence_dir}"
            )
        if errors:
            raise ValueError(
                f"TAPO evidence artifact contract violation for {clean_date}:\n" +
                "\n".join(f"  - {err}" for err in errors)
            )

    result = {
        "date": clean_date,
        "evidence_dir": str(evidence_dir),
        "feeding_detected": feeding_detected,
        "has_summary": has_summary,
        "has_timeline": has_timeline,
        "feeding_phases_count": len(feeding_phases),
        "source_clips_count": len(source_clips),
        "source_clips": [c.name for c in source_clips]
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate TAPO evidence artifact bundle before upload.")
    parser.add_argument("--evidence-dir", type=str, default="/tmp/output", help="Directory containing staged evidence")
    parser.add_argument("--date", type=str, required=True, help="Target date YYYYMMDD")
    parser.add_argument("--strict-feeding", action="store_true", help="Fail if no feeding was detected")
    args = parser.parse_args()

    try:
        res = validate_tapo_evidence_bundle(
            evidence_dir=Path(args.evidence_dir),
            target_date=args.date,
            strict_feeding_required=args.strict_feeding
        )
        print(f"✅ TAPO evidence artifact bundle validated for {args.date}:")
        print(f"   Feeding detected : {res['feeding_detected']}")
        print(f"   Summary present  : {res['has_summary']}")
        print(f"   Timeline present : {res['has_timeline']} ({res['feeding_phases_count']} phases)")
        print(f"   Source clips ({res['source_clips_count']}): {', '.join(res['source_clips']) or 'None (genuine no-feeding)'}")
        sys.exit(0)
    except Exception as e:
        print(f"❌ TAPO evidence artifact validation FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
