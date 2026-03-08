# Current Tasks

## Raspberry Pi 5: 24/7 Monitoring Setup

### Working Now ✅
- [x] RTSP camera connection (TCP transport)
- [x] Motion detection via `test_motion_pi.py` (MOG2 background subtraction)
- [x] Video recording with 3-second pre-buffer
- [x] Google Drive upload via `rclone bisync` (background sync)
- [x] Frame-based motion trigger (no ONVIF dependency needed)

### Current Recommendation: Use `motion_recorder.py`
For 24/7 Pi 5 monitoring:
```bash
cd /home/pi5/Feeder/fair-feeder
source .venv/bin/activate
nohup python motion_recorder.py > motion.log 2>&1 &
```

**What it does:**
- Detects motion (MOG2) ✅
- Records video ✅
- Runs YOLOv8n (at 0.10 conf) on video frames to find cats ✅
- Uploads CAT videos to Google Drive ✅
- Deletes NON-CAT videos locally ✅

### Solved: Cat Detection on Pi 5
- [x] **Fix cat detection in `motion_recorder.py`** 
  - Abandoned `ai-edge-litert` (EfficientDet) due to bad inference + hallucinating objects (ovens, sinks).
  - Switched to `ultralytics` YOLOv8n which easily detects partial cats at ground-level view using 0.10 confidence.

### To Deploy: 24/7 Systemd Service
- [ ] Create systemd timer to run `test_motion_pi.py` at boot
- [ ] Monitor rclone uploads via log file
- [ ] Set up storage cleanup script (delete uploaded videos)
- [ ] Monitor disk usage: `df -h /home/pi5/Pictures/gdrive-randomdice-sync/`

### For Future: Colab Batch Analysis
- [ ] Download recorded videos from Google Drive
- [ ] Run smoketest.ipynb pipeline (YOLOv11 analysis)
- [ ] Generate feeding summaries + Telegram reports
- [ ] Scale when Pi recordings become routine

---

## Completed (Archived)
- [x] Raspberry Pi deployment: `systemd` auto-start service
- [x] Background `rclone bisync` trigger
- [x] OS path detection (Windows vs Pi)
- [x] Smoketest pipeline (3-stage: cache → analytics → output)
- [x] YOLOv11 V13 model trained (Colab)
- [x] Telegram bot integration (Colab)
- [x] Motion-triggered recording via ONVIF (deprecated; test_motion_pi.py is simpler)
