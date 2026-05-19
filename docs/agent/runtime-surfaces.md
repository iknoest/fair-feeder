# Runtime Surfaces

Fair Feeder has three active runtime surfaces: Pi 5 recording, GitHub Actions
reporting, and Colab/Kaggle interactive analysis.

## Raspberry Pi 5

Production stack:

`motion_recorder.py` on systemd -> MOG2 motion detection -> YOLOv8n cat filter ->
rclone copy to Drive -> `sync_cleanup.sh` deletes local videos older than 3 days.

Telegram commands:
- `/status`
- `/lastclip`
- `/weight`
- `/help`

Weight tracking:
- `/weight` opens an inline menu: Log Weight, History, Edit.
- Logs save to `weight_log.csv` in `DRIVE_OUTPUT_DIR`.
- Pi syncs the file to Drive through rclone.
- Morning report sends a 30-day reminder if no weight was logged.
- `/syncstatus` was merged into `/status`; Drive file count is appended with
  `rclone size` and an 8-second timeout.

## Pi Deployment Expectation

When changing `motion_recorder.py` or other Pi-runtime files, deploy and restart
the Pi service unless the user explicitly says not to.

Standard flow:
1. SCP changed files to `/home/pi5/Feeder/fair-feeder/`.
2. Run Pi-side compile check, usually:
   `cd /home/pi5/Feeder/fair-feeder && ./.venv/bin/python -m py_compile motion_recorder.py`
3. Restart `cat-monitor.service`.
4. Verify `systemctl is-active cat-monitor.service`.
5. Check recent logs/status for immediate failures.

## Pi Runtime Constraints

- RTSP must use TCP transport; UDP is unreliable on Pi 5.
- YOLOv8n cat filter runs at confidence 0.10.
- Older EfficientDet hallucinated boxes at the ground-level camera angle.
- `infisical-sdk` has no ARM64 wheel; use Infisical Universal Auth REST via
  `requests` on Pi.
- `ai-edge-litert` API has been unstable on Pi; YOLOv8n is more reliable.
- Drive sync uses `rclone copy`, not `bisync`, to avoid `.lck` deadlocks.
- 2304x1296 at 15 fps is roughly 50-100 MB/min; save only cat-positive clips.

## GitHub Actions

GitHub Actions runs `morning_report.ipynb` via papermill.

Current schedule:
- Cron: `23 0 * * *` UTC.
- Reason: GitHub scheduled workflows can start hours late; observed delays were
  about 4h10m on 2026-05-18 and 2026-05-19.
- The minute is intentionally not `0` because GitHub documents high load at the
  start of every hour.
- The workflow waits until 06:35 Europe/Amsterdam if it starts early.
- Scheduler heartbeat belongs in GitHub summaries and `feeding_log.csv`, not in
  Telegram.

## Colab / Kaggle

- `smoketest.ipynb` is for interactive threshold tuning.
- `batch_review.ipynb` is for historical reprocessing.
- Training notebooks run on Colab/Kaggle T4.
- CI is for daily automation; Colab is for interactive analysis and archival work.

## Task Split

| Task | Pi 5 | Colab / CI |
|------|------|------------|
| 24/7 motion recording | yes | no |
| YOLOv11 analysis | no, CPU-bound | yes |
| Daily report generation | no | yes |

