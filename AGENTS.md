# AGENTS.md - Fair Feeder

Fair Feeder is a computer-vision cat feeding monitor for Dan and Sanbo. It uses a
Tapo C210 IR camera, a Logitech C925e USB camera, YOLOv11, Tapo timestamp OCR,
Google Drive, GitHub Actions, and Telegram reports to determine whether Dan ate
enough and whether Sanbo stole food.

This file is the compact bootstrap. Load the detailed docs below only when the
task needs them.

## Document Reference Index

- Project map, file layout, dependencies, services:
  `docs/agent/project-map.md`
- Pi runtime, GitHub Actions, Colab/CI roles:
  `docs/agent/runtime-surfaces.md`
- `morning_report.ipynb`, FeedingTracker, snapshots, Telegram output:
  `docs/agent/morning-report-pipeline.md`
- CI failures, scheduler delay, Drive, `feeding_log.csv`:
  `docs/agent/ci-drive-runbook.md`
- V14/V15, Roboflow, mAP, training decisions:
  `docs/agent/model-data-flywheel.md`
- Coding, notebook, YOLO, OCR, and verification conventions:
  `docs/agent/coding-conventions.md`
- Active design decisions and failure patterns:
  `docs/agent/decisions-and-gotchas.md`
- Documentation update rules:
  `docs/agent/documentation-policy.md`
- Suggested `.claude/skills/` and slash-command breakdown:
  `docs/agent/agent-workflows.md`

## Task Routing

- For GitHub Actions, Drive, scheduler, or `feeding_log.csv`, read
  `docs/agent/ci-drive-runbook.md` first.
- For `morning_report.ipynb`, snapshots, FeedingTracker, or Telegram reports,
  read `docs/agent/morning-report-pipeline.md` first.
- For `motion_recorder.py`, Pi, systemd, RTSP, or rclone, read
  `docs/agent/runtime-surfaces.md` first.
- For model, Roboflow, mAP, flags, or retraining, read
  `docs/agent/model-data-flywheel.md` first.
- For code or notebook edits, read `docs/agent/coding-conventions.md` first.
- For lessons or context-file updates, read `docs/agent/documentation-policy.md`
  first.

## Non-Negotiable Project Rules

- Use actual class names: Dan, Sanbo, Dan_hand, Bowl, Kibble.
- Never hardcode secrets, Chat IDs, folder IDs, or environment-specific tokens.
- Maintain independent systemd services and sync folders for multi-camera setups
  to prevent resource contention.
- Files imported by `morning_report.ipynb` or `motion_recorder.py`, or hardcoded
  in workflows/services, stay at repository root.
- YOLO calls preserve 16:9 with `rect=True`; `imgsz` is a single int.
- Use `model.names` for class mapping; do not assume class index order.
- Annotated video is boxes-only with `show_label=False`.
- `.ipynb` edits are JSON edits; strip `\r` from cell source on Windows.
- Use `tqdm.auto`, not `tqdm.notebook`, in CI/notebook code.
- Tapo OCR replacement order matters: `\|:` -> `:1` before `\|` -> `1`.
- Stitch clips only when the gap is 10 seconds or less.
- Scheduler heartbeat belongs in GitHub summaries and `feeding_log.csv`, not in
  Telegram.
- Morning-report cron is intentionally early and not at minute 0; GitHub has
  delayed scheduled workflows by about 4h10m.
- `feeding_log.csv` is updated through Drive `update()`, never CI `create()`.
- Large binary archives are not uploaded from CI; Colab handles archival.

## Kibble Snapshot Invariants

- Prefer a stable no-cat Kibble snapshot before cats arrive at the bowl.
- `kibble_dispensed` must keep collecting clean post-hand/pre-arrival frames until
  cat arrival or timeout.
- If Dan_hand is absent or filtered out, still emit a generic
  `kibble_dispensed` snapshot from the best clean pre-cat frame.
- Do not send `kibble_start` snapshots.

## Runtime Expectations

- Pi-runtime changes should be deployed to the Pi unless the user says not to:
  SCP, Pi-side compile, restart `cat-monitor.service`, verify active status and
  recent logs.
- CI-facing or Pi-runtime fixes should be verified, reviewed for secrets and
  unrelated work, committed, and pushed to `main` when safe.
- If a push would include destructive history changes, credentials, or unrelated
  user work, stop and ask first.

## Communication

- Be concise: say what changed and why.
- Show actual vs expected behavior when reporting bugs.
- Reference mAP50 and per-class AP50 for model-quality discussions.
- Preserve project-specific gotchas by moving details into `docs/agent/*`, not by
  deleting them.

