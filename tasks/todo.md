# Current Tasks

## In Progress
- [ ] Testing model on more real-world videos (owner ran 2 so far)
- [ ] Evaluating detection quality against 8-scenario checklist

## Planned Next (prioritized)
1. Run 5–10 more test videos across different scenarios (IR, motion blur, two cats)
2. Evaluate if model needs retraining with more examples
3. Automate video-to-analysis pipeline (currently manual notebook runs)

## Recently Completed
- [x] Raspberry Pi deployment: `systemd` auto-start service (`cat-monitor.service`)
- [x] Background `rclone bisync` trigger inside `motion_recorder.py` for headless Drive sync
- [x] OS path detection (Windows vs Pi) for output directories
- [x] Smoketest pipeline reorganized into 3 stages (YOLO cache → analytics → output)
- [x] Detection cache stores compressed JPEG frames (~50KB/frame)
- [x] Motion-triggered recording via ONVIF events (`motion_recorder.py`)
- [x] ONVIF debounce handling, duration naming, cat detection filter
- [x] Credentials sanitized for Git (placeholders + `README_GIT_PULL.md`)
- [x] CLAUDE.md restructured with workflow orchestration + lessons learned
