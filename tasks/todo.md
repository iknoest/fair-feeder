# Current Tasks

---

## Pipeline Regressions — ALL VERIFIED ✅

All 4 bugs fixed and verified in CI run 2026-03-26.

### Bug 1 — FeedingTracker returns 0 (issue #31) ✅ VERIFIED
### Bug 2 — Wrong annotated video sent to Telegram (issue #32) ✅ VERIFIED
### Bug 3 — Drive video upload dropped by design (issue #33) ✅ RESOLVED
### Bug 4 — feeding_log.csv not accumulating (issue #34) ✅ VERIFIED

---

## Phase B: Fix + Communicate + Trends

### B1 — Bug Fixes ✅
- [x] Fix hardcoded camera password in `config.py:48`
- [x] Fix RTSP reconnect in `motion_recorder.py` — TCP transport

### B2 — Pi ↔ Telegram Two-Way Health Check
- [x] Add `TelegramCommandListener` class to `motion_recorder.py`
- [x] Wire into main loop (daemon thread, shares `RecordingController` instance)
- [x] `/weight` inline menu with Log/History/Edit buttons; weight chart X-axis fixed (integer ticks + MM-DD labels)
- [x] `/syncstatus` merged into `/status` — Drive file count appended inline
- [x] Add 24/7 bowl-position alert using YOLOv8n COCO `bowl`
- [x] **Test on Pi**: `/status` verified with bowl count/centered state after Pi deployment
- [ ] Verify `/lastclip`, `/weight`, `/help` from phone after latest Pi deployment

### B3 — Morning Kibble Report (GitHub Actions) ✅
- [x] Service account auth + Drive sharing
- [x] CI-compatible notebook (`RUNNING_IN_CI` guard, `tqdm.auto`)
- [x] Cron schedule running with early UTC compensation (`0 3 * * *`) plus wait-until-06:35 Amsterdam guard
- [x] Scheduler heartbeat added to Telegram + GitHub Actions summary
- [x] Merged clip detection and stitching working in Phase 1
- [x] Merged CI clips send one report set; later clips skipped once a real feeding event is captured
- [x] Timeline chart generating correctly from detection cache
- [x] **Verify all fixes work on real morning clips** — CI run 2026-03-26 all green ✅

### B4 — Weekly Trend Digest
~~Dropped~~ — user sees no value. Daily report already answers the question. Removed from scope.

---

## Phase C: Data Flywheel — Continuous Model Improvement
*(spec: `docs/superpowers/specs/2026-03-26-data-flywheel-design.md`)*

### C1 — Auto-flagging + Roboflow upload (daily CI pipeline) ✅ VERIFIED
- [x] Remove Drive video upload from CI (fixes SA 403 quota issue)
- [x] Implement Phase 2.5: auto-flag suspicious detections from cache (`flagging.py`)
- [x] Implement Phase 2.6: upload flagged frames to Roboflow via SDK (`roboflow_upload.py`)
- [x] Add flag summary to Telegram report
- [x] Log daily flag trend fields to `feeding_log.csv` for monthly retraining direction
- [x] Extract flagging + upload into shared modules
- [x] Add `ROBOFLOW_API_KEY` + `ROBOFLOW_WORKSPACE` to CI workflow env
- [x] **User setup**: GitHub secrets added (ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE) ✅
- [x] **User setup**: Colab Secrets added (ROBOFLOW_API_KEY) ✅
- [x] **Verify**: CI run 2026-03-26 — 42 frames flagged, all uploaded to Roboflow ✅

### C2 — Batch reprocessing notebook ✅
- [x] Create `batch_review.ipynb` for Colab (reprocess historical Pi videos)
- [x] Add FEEDING_WINDOW_ONLY toggle and MAX_VIDEOS cap
- [x] Run 1: FEEDING_WINDOW_ONLY=True — 19 feeding-window videos processed, 263 frames flagged
- [x] Upload flagged frames to Roboflow with pre-annotations — 231 frames uploaded
- [x] Add EXCLUDE_DATES filter and upload deduplication via tracking file

### C3 — First retrain cycle ✅ DEPLOYED
- [x] Review flagged frames in Roboflow, correct labels (775 images, 5949 annotations)
- [x] Generate dataset v14, retrain on Kaggle (YOLOv11s, copy_paste=0.0)
- [x] Compare: mAP50 0.956→0.957, Sanbo AP50 0.959→0.985, mAP50-95 0.739→0.754
- [x] V14 deployed to CI — first run 2026-03-28: 6 flags (vs 10-20+ with v13), zero blip-sanbo
- [x] Update `GDRIVE_MODEL_FILE_ID` GitHub secret to point to v14 model
- [x] Monitor v14 daily flag counts before next retrain cycle; April flagged frames reviewed for V15 candidate

### C4 — V15 candidate review
- [x] Train V15 candidate with 155 manually revised April flagged images
- [x] Compare V15 against pasted V13 validation metrics
- [x] Re-run V14 in `smoketest.ipynb` and document that historical metrics differ from current validation
- [ ] Re-run V14 and V15 on the same fixed holdout dataset before deployment decision
- [ ] Decide whether to deploy V15 or keep V14 and monitor May flags

### C-later — Nice to have
- [x] Feeding window filter — `smoketest.ipynb` filters clips to 06:18–06:30; multi-clip stitch via ffmpeg concat
- [x] Reduce noisy `kibble-jump-*` flags by only comparing clear bowl frames
- [x] 24/7 bowl-position alert in `motion_recorder.py` using YOLOv8n COCO `bowl`
- [ ] Optional Bowl ROI zone tuning in `motion_recorder.py` if YOLOv8n centered-count is noisy
- [ ] Lightweight Dan/Sanbo classifier on Pi — tag clip filenames with identity
- [ ] Telegram-interactive flagging (reply to report to flag issues)

---

## Ongoing
- [ ] Monitor rclone uploads via log file
- [ ] Monitor disk usage: `df -h /home/pi5/Pictures/gdrive-randomdice-sync/`
- [ ] B2 Pi test: verify `/lastclip`, `/weight`, `/help` from phone

---

## Completed (Archived)
- [x] CI fixes: ffmpeg in workflow; timezone filter; stitch; FPS remux; empty bowl exit disabled
- [x] Raspberry Pi deployment: systemd service with Telegram ping
- [x] OS path detection (Windows vs Pi)
- [x] Smoketest pipeline (3-stage: cache → analytics → output)
- [x] YOLOv11 V14 model trained (mAP50=0.957, Sanbo AP50=0.985)
- [x] Telegram bot integration (Colab + CI)
- [x] Motion-triggered recording (MOG2)
- [x] RTSP camera connection (TCP transport)
- [x] Video recording with 3s pre-buffer
- [x] Google Drive upload via rclone (background sync)
- [x] Telegram boot alerts via Infisical REST API (ARM64 workaround)
- [x] Cat detection filter (YOLOv8n at 0.10 conf)
- [x] Systemd service (`cat-monitor.service`)
- [x] Infisical secret management
- [x] Annotated video compression (ffmpeg crf=28, 720p)
- [x] Per-video Telegram send inside processing loop
