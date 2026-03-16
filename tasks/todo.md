# Current Tasks

## Phase B: Fix + Communicate + Trends
*Design doc: `docs/plans/2026-03-08-system-improvements-design.md`*

### B1 — Bug Fixes
- [x] Fix hardcoded camera password in `config.py:48` — placeholder + Infisical/env var load
- [x] Fix RTSP reconnect in `motion_recorder.py` — TCP transport already used in reconnect path

### B2 — Pi ↔ Telegram Two-Way Health Check
- [x] Add `TelegramCommandListener` class to `motion_recorder.py`
- [x] Wire into main loop (daemon thread, shares `RecordingController` instance)
- [ ] **Test on Pi**: send `/status`, `/lastclip`, `/help` from phone and verify replies

### B3 — Morning Kibble Report (GitHub Actions)
- [x] Create Google Cloud project + service account (one-time)
- [x] Share Drive folder with service account email
- [x] Store JSON key as GitHub Actions secret (`GDRIVE_SERVICE_ACCOUNT_KEY`)
- [x] Add environment-detection auth cell to `smoketest.ipynb` (Colab vs CI)
- [x] Add CSV append cell to `smoketest.ipynb` (writes to `feeding_log.csv` on Drive)
- [x] Create `.github/workflows/morning-report.yml`
  - Cron: `45 23 * * *` (= 6:45am Thailand UTC+7)
  - Install packages (ultralytics, easyocr, opencv, papermill, google-auth)
  - Download new Drive videos via service account
  - Run `smoketest.ipynb` via papermill
  - Upload results back to Drive
- [x] Fix: guard Colab-only cells with `RUNNING_IN_CI` check
- [x] Fix: replace `tqdm.notebook` with `tqdm.auto` for CI compatibility
- [x] Fix: wrap `feeding_log.csv` create in try/except for quota error on first run
- [x] Fix: pass `GDRIVE_MODEL_FILE_ID` secret to papermill step in workflow
- [x] Test end-to-end — CI pipeline runs successfully

### B4 — Weekly Trend Digest
- [ ] Add second job to `morning-report.yml`
  - Cron: `0 0 * * 1` (= 7:00am Monday Thailand time)
  - Reads last 7 rows of `feeding_log.csv`
  - Sends weekly Telegram digest
- [ ] Verify CSV schema matches digest format

---

## Phase C: Smarter Recording + Data Flywheel
*(Start after Phase B is stable and running for 1–2 weeks)*

- [x] Feeding window filter in morning report — `smoketest.ipynb` filters clips to 06:18–06:30 by filename timestamp; multi-clip stitch via ffmpeg concat
- [ ] Bowl ROI zone filter in `motion_recorder.py` (`BOWL_ROI` in `config.py`)
- [ ] Lightweight Dan/Sanbo classifier on Pi — tag clip filenames
- [ ] `/review` Telegram command — sends flagged low-confidence clips
- [ ] Auto-copy confirmed clips to `training_candidates/` folder on Drive

---

## Ongoing
- [ ] Monitor rclone uploads via log file
- [ ] Monitor disk usage: `df -h /home/pi5/Pictures/gdrive-randomdice-sync/`

---

## Completed (Archived)
- [x] CI fixes: ffmpeg installed in workflow; feeding window filter (06:18–06:30) + multi-clip stitch in `smoketest.ipynb`; `motion_recorder.py` stream FPS + remux fix
- [x] Raspberry Pi deployment: `systemd` auto-start service with Telegram ping
- [x] Background `rclone bisync` trigger
- [x] OS path detection (Windows vs Pi)
- [x] Smoketest pipeline (3-stage: cache → analytics → output)
- [x] YOLOv11 V13 model trained (Colab)
- [x] Telegram bot integration (Colab)
- [x] Motion-triggered recording via ONVIF (deprecated; motion_recorder.py is simpler)
- [x] RTSP camera connection (TCP transport)
- [x] Motion detection via `motion_recorder.py` (MOG2 background subtraction)
- [x] Video recording with 3-second pre-buffer
- [x] Google Drive upload via `rclone bisync` (background sync)
- [x] Telegram boot alerts via Infisical REST API (ARM64 SDK workaround)
- [x] Cat detection filter (YOLOv8n at 0.10 conf)
- [x] Systemd service (`cat-monitor.service`)
