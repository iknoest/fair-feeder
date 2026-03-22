# Current Tasks

---

## Pipeline Regressions — FIXED (awaiting morning verification)

All 4 bugs fixed in `morning_report.ipynb` (commits `ae49d5c`–`c83fcd4`). Awaiting next scheduled run (06:45 CET) to verify with real clips.

### Bug 1 — FeedingTracker returns 0 (issue #31) ✅ FIXED
- [x] Root cause: `_find_clear_kibble_count` searches no-cat frames; model only detects kibble with cats present
- [x] Fix: added `_find_kibble_at_phase_entry` / `_find_kibble_at_phase_exit` fallback methods to FeedingTracker
- [ ] **Verify**: next morning run shows non-zero kibble start, % per cat

### Bug 2 — Wrong annotated video sent to Telegram (issue #32) ✅ FIXED
- [x] Root cause: Phase 1/2 re-scanned SOURCE_DIR overwriting video_paths from stitch cell
- [x] Fix: guarded re-scan behind `if not RUNNING_IN_CI:` in Cells 12 and 13
- [x] Fix: merged_names now uses `merged_sources` dict (set by Cell 1)
- [ ] **Verify**: next morning run sends merged annotated video

### Bug 3 — Annotated video not uploaded to Drive (issue #33) ✅ FIXED
- [x] Root cause: annotated video path used SOURCE_DIR; file is written to OUTPUT_DIR
- [x] Fix: Cell 14 now uses `OUTPUT_DIR/{stem}_annotated.mp4`
- [ ] **Verify**: annotated video appears in Drive `Test_postmodel_output/` after next run

### Bug 4 — feeding_log.csv not accumulating (issue #34) ✅ FIXED
- [x] Root cause: CSV cell ran before Phase 1–3 (summary undefined) + wrong video path
- [x] Fix: CSV cell moved to after Phase 3; reads from `video_results[-1]['summary']`
- [x] Fix: cell now skips entirely when no videos were processed (no zero-row pollution)
- [ ] **Verify**: CSV has new row with real kibble values after next morning run

---

## Phase B: Fix + Communicate + Trends

### B1 — Bug Fixes ✅
- [x] Fix hardcoded camera password in `config.py:48`
- [x] Fix RTSP reconnect in `motion_recorder.py` — TCP transport

### B2 — Pi ↔ Telegram Two-Way Health Check
- [x] Add `TelegramCommandListener` class to `motion_recorder.py`
- [x] Wire into main loop (daemon thread, shares `RecordingController` instance)
- [ ] **Test on Pi**: send `/status`, `/lastclip`, `/help` from phone and verify replies

### B3 — Morning Kibble Report (GitHub Actions)
- [x] Service account auth + Drive sharing
- [x] CI-compatible notebook (`RUNNING_IN_CI` guard, `tqdm.auto`)
- [x] Cron schedule running (`45 5 * * *` = 06:45 CET)
- [x] Merged clip detection and stitching working in Phase 1
- [x] Timeline chart generating correctly from detection cache
- [ ] **Verify all fixes work on real morning clips** — next scheduled run 06:45 CET

### B4 — Weekly Trend Digest
~~Dropped~~ — user sees no value. Daily report already answers the question. Removed from scope.

---

## Phase C: Smarter Recording + Data Flywheel
*(start only after B3 pipeline is stable and verified for 1–2 weeks)*

- [x] Feeding window filter — `smoketest.ipynb` filters clips to 06:18–06:30; multi-clip stitch via ffmpeg concat
- [ ] Bowl ROI zone filter in `motion_recorder.py` (`BOWL_ROI` in `config.py`)
- [ ] Lightweight Dan/Sanbo classifier on Pi — tag clip filenames with identity
- [ ] `/review` Telegram command — sends flagged low-confidence clips
- [ ] Auto-copy confirmed clips to `training_candidates/` folder on Drive

---

## Ongoing
- [ ] Monitor rclone uploads via log file
- [ ] Monitor disk usage: `df -h /home/pi5/Pictures/gdrive-randomdice-sync/`
- [ ] B2 Pi test: verify Telegram commands work from phone

---

## Completed (Archived)
- [x] CI fixes: ffmpeg in workflow; timezone filter; stitch; FPS remux; empty bowl exit disabled
- [x] Raspberry Pi deployment: systemd service with Telegram ping
- [x] OS path detection (Windows vs Pi)
- [x] Smoketest pipeline (3-stage: cache → analytics → output)
- [x] YOLOv11 V13 model trained (mAP50=0.928 for Dan_hand)
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
