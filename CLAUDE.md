# CLAUDE.md - Fair Feeder

Fair Feeder monitors Dan and Sanbo feeding from Tapo C210 video. The production
pipeline uses YOLOv11, OCR of the Tapo timestamp, GitHub Actions, Google Drive,
and Telegram reports.

Use this file as a short bootstrap. Load the referenced docs only when relevant.

## Critical Shortcuts

- Git status in this repo:
  `git -c safe.directory=C:/Users/AVAVAVA/.gemini/antigravity/scratch/fair-feeder status --short`
- Notebook JSON check:
  `python -c "import json; json.load(open('morning_report.ipynb', encoding='utf-8'))"`
- Focused report behavior test on this machine:
  `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; C:\Users\AVAVAVA\anaconda3\python.exe -m pytest tests/test_daily_report_behavior.py -q`
- Pi compile check after `motion_recorder.py` edits:
  `cd /home/pi5/Feeder/fair-feeder && ./.venv/bin/python -m py_compile motion_recorder.py`

## Environment

- GitHub Actions runs `morning_report.ipynb` through papermill.
- Current cron is `0 2 * * *` UTC, then wait until 06:35 Europe/Amsterdam if the
  runner starts early.
- Secrets come from Infisical or GitHub Actions env. Never hardcode values.
- Common secret/env names: `INFISICAL_ID`, `INFISICAL_SECRET`,
  `INFISICAL_PROJECT_ID`, `TelegramBotToken`, `TelegramChatId`,
  `TELEGRAM_CHAT_ID`, `TAPO_IP`, `TAPO_USER`, `TAPO_PASS`,
  `GDRIVE_UPLOAD_FOLDER_ID`, `GDRIVE_MODEL_FILE_ID`, `GITHUB_TOKEN`.

## Document Reference Index

- Project map, root file rules, services:
  `docs/agent/project-map.md`
- Pi runtime, GitHub Actions, Colab/CI:
  `docs/agent/runtime-surfaces.md`
- Morning report, FeedingTracker, snapshots, Telegram:
  `docs/agent/morning-report-pipeline.md`
- CI, Drive, scheduler, `feeding_log.csv`:
  `docs/agent/ci-drive-runbook.md`
- Models, Roboflow, mAP, data flywheel:
  `docs/agent/model-data-flywheel.md`
- Code, notebook, YOLO, OCR conventions:
  `docs/agent/coding-conventions.md`
- Decisions and failure patterns:
  `docs/agent/decisions-and-gotchas.md`
- Documentation update policy:
  `docs/agent/documentation-policy.md`
- Skill and slash-command candidates:
  `docs/agent/agent-workflows.md`

## Rules Always Loaded

- Use class names Dan, Sanbo, Dan_hand, Bowl, Kibble.
- Root dependencies stay at root if imported by notebooks/runtime or hardcoded in
  workflows/services.
- YOLO calls use `rect=True`; `imgsz` is a single int; class mapping uses
  `model.names`.
- Annotated video is boxes-only with `show_label=False`.
- Edit `.ipynb` files as JSON and strip `\r` from cell source on Windows.
- Use `tqdm.auto`, not `tqdm.notebook`.
- Tapo OCR replacement order: `\|:` -> `:1`, then `\|` -> `1`.
- `feeding_log.csv` uses Drive `update()`, never CI `create()`.
- Scheduler heartbeat belongs in GitHub summary and `feeding_log.csv`, not
  Telegram.
- Large CI binaries are not uploaded to Drive; Colab archives.
- For Pi runtime changes: SCP, Pi-side compile, restart `cat-monitor.service`,
  verify active status and recent logs.
- For CI/Pi fixes: verify, review diff for secrets/unrelated work, commit, and
  push to `main` when safe.

## Kibble Snapshot Rules

- Prefer stable no-cat Kibble before cats arrive.
- Keep collecting clean post-hand/pre-arrival frames until cat arrival or timeout.
- If Dan_hand is missing or filtered out, still emit generic `kibble_dispensed`
  from the best clean pre-cat frame.
- Do not send `kibble_start`.

