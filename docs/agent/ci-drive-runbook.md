# CI and Drive Runbook

Use this for GitHub Actions, Google Drive, `feeding_log.csv`, scheduler delay, and
CI notebook failures.

## Current Schedule & Delivery Architecture

- Primary delivery trigger: Pi watchdog directly dispatches GitHub Actions at 07:15 Europe/Amsterdam once drain completes.
- Tertiary remote recovery cron: `0 8 * * *` with `timezone: 'Europe/Amsterdam'`.
- Workflow structure: Deterministic DAG (`prepare` -> `tapo-report` -> `logitech-report`).
  - `prepare` resolves `target_date` once (YYYYMMDD).
  - `tapo-report` runs with concurrency `fair-feeder-${TARGET_DATE}-TAPO`, runs `morning_report.ipynb`, exports `tapo_timeline_${TARGET_DATE}.json` as an artifact and to Drive.
  - `logitech-report` runs with concurrency `fair-feeder-${TARGET_DATE}-LOGITECH`, downloads `tapo_timeline_${TARGET_DATE}.json`, and performs cross-camera reconciliation.
- Exactly-once delivery: Durable ledger on Google Drive (`delivery_ledger_${TARGET_DATE}_${CAMERA}.json`) tracks item-level delivery (`summary`, `timeline`, `snap_*`, `video_*`).
  - Preflight checks `camera_fully_delivered` and exits in < 2s on duplicate triggers.
  - Partial delivery failure resumes cleanly without duplicate messages.
- Scheduler delay and heartbeat belong in GitHub summaries and `feeding_log.csv`, not in Telegram.

## CI Preflight Checklist

Before pushing a GitHub Actions fix:

- Every secret read by the notebook is listed under the workflow step `env:`.
- Headless imports work; no `tqdm.notebook`, `ipywidgets`, or unguarded
  `google.colab`.
- Colab-only imports and `drive.mount()` are guarded by `RUNNING_IN_CI`.
- Service account uses Drive `update()`, not `create()`.
- Datetime filtering uses `pytz.timezone('Europe/Amsterdam')`, never naive UTC.
- System dependencies such as ffmpeg are installed before pip if required.
- Execute the notebook locally with `jupyter nbconvert --execute` or `papermill`
  when the change affects notebook runtime.

When a CI run fails, list every likely failure mode before pushing. Do not
fix-push-wait-fail one issue at a time when several causes are visible.

## Google Drive Policy

- Archive pipeline outputs per run with timestamped filenames.
- Never overwrite one canonical archive file.
- Large binary outputs are not uploaded from CI because the service account has no
  storage quota.
- Colab with the user account handles archive uploads.
- Telegram already delivers daily outputs.

## `feeding_log.csv`

`feeding_log.csv` is the exception to the timestamped archive rule.

Required behavior:
1. Download the current Drive file with `get_media()`.
2. Remove today's row if present.
3. Append the fresh row.
4. Use Drive `update()`.
5. Do not call Drive `create()` from CI.

Current columns:

`date, camera, dan_kibble, sanbo_kibble, hand_feeding, compensation, video_count,
dan_first_arrival, sanbo_first_arrival, schedule_time, start_time,
flagged_frames, roboflow_uploaded, roboflow_skipped, roboflow_failed,
flag_top_tags, dan_weight, sanbo_weight`

Notes:
- `schedule_time` and `start_time` are Europe/Amsterdam local time with DST.
- Weight columns are backfilled from Pi-generated `weight_log.csv` in
  `GDRIVE_UPLOAD_FOLDER_ID`.
- Schedule/start columns are backfilled from GitHub Actions run history when
  `GITHUB_TOKEN` is available.

## Known CI Failure Patterns

- Missing env var in workflow despite secret existing in repo settings.
- Drive `403 storageQuotaExceeded` from service account `create()`.
- `tqdm.notebook` or widgets failing under papermill.
- Naive UTC date filtering missing morning captures.
- Phase 1/2 rescanning `SOURCE_DIR` and overwriting stitched output.
- Notebook JSON edited on Windows with stray `\r`, causing IPython lexer errors.
