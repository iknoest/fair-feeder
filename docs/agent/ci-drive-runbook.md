# CI and Drive Runbook

Use this for GitHub Actions, Google Drive, `feeding_log.csv`, scheduler delay, and
CI notebook failures.

## Current Schedule Behavior

- Workflow cron is `0 2 * * *` UTC.
- GitHub scheduled workflows can start hours after cron.
- The workflow waits until 06:35 Europe/Amsterdam if it starts early.
- `schedule_time` and `start_time` are recorded in Europe/Amsterdam local time.
- Scheduler heartbeat belongs in GitHub summaries and `feeding_log.csv`, not in
  Telegram.

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

`date, dan_kibble, sanbo_kibble, hand_feeding, compensation, video_count,
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

