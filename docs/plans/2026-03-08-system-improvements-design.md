# Fair Feeder — System Improvements Design
**Date:** 2026-03-08
**Phase:** B (Fix + Communicate + Trends) → C (Smarter Recording + Data Flywheel)

---

## Context

Phase 1 (Pi 5 motion recording + Colab analysis + Telegram) is working. This design
covers the next two phases of improvements to make the system reliable, observable,
and self-improving.

---

## Phase B: Fix + Communicate + Trends

### B1 — Bug Fixes (non-negotiable)

| Fix | File | Change |
|-----|------|--------|
| Hardcoded camera password in fallback | `config.py:48` | Replace real password string with `<YOUR_CAMERA_PASSWORD>` |
| RTSP reconnect uses UDP (undoes TCP fix) | `motion_recorder.py:257` | Append `?rtsp_transport=tcp` on reconnect |

> rclone bisync stays as-is — single authorized folder, user intentionally uses
> two-way sync to clean up Pi by deleting from Drive side.

### B2 — Pi ↔ Telegram Two-Way Health Check

**Goal:** User can query live Pi status from phone at any time.

**Implementation:** New `TelegramCommandListener` class added to `motion_recorder.py`.
Runs in a daemon thread, polls Bot API every 2 seconds, dispatches to handlers.
`RecordingController` instance passed in so handlers can read live stats.

**Commands:**
- `/status` — uptime, clips saved today, clips deleted today, disk space on
  recordings folder, last motion detected time
- `/lastclip` — sends most recent saved cat clip as Telegram video
- `/help` — lists available commands

**Scope:** Read-only. No remote start/stop of recording via Telegram.

**Secret loading:** Uses existing Infisical REST API pattern already in
`motion_recorder.py` — no new auth mechanism.

### B3 — Morning Kibble Report via GitHub Actions

**Trigger timing:** 6:20am auto-feeder fires → cat eats (~4 min) → Pi saves
clip → rclone uploads to Drive. GitHub Actions cron fires at **6:45am Thailand
time** (`45 23 * * *` UTC).

**Flow:**
```
GitHub Actions (cron 6:45am)
  → Authenticate Google Drive via service account
  → Find videos in Drive folder newer than last run timestamp
  → Download to runner temp directory
  → Run smoketest.ipynb via papermill (zero notebook migration)
  → Save results + updated CSV back to Drive
  → Send Telegram morning report
```

**Service account setup (one-time):**
1. Create Google Cloud project (free)
2. Create service account, download JSON key
3. Share Drive folder with service account email
4. Store JSON key as GitHub Actions secret (`GDRIVE_SERVICE_ACCOUNT_KEY`)
5. Add auth cell at top of smoketest.ipynb (detects environment: Colab vs CI)
6. Add CSV append cell at bottom of smoketest.ipynb

**What does NOT change in smoketest.ipynb:**
- All analysis cells (YOLOv11, EasyOCR, FeedingTracker)
- All thresholds and config
- Telegram send logic
- Class names and detection parameters

**Environment detection pattern in notebook:**
```python
import os
if os.getenv('GITHUB_ACTIONS'):
    # Service account auth
else:
    # Colab drive.mount() as before
```

**Colab stays as the manual testing/debugging environment** — nothing changes
for interactive use.

### B4 — Long-Term Trend Tracking

**Storage:** `feeding_log.csv` on Google Drive, appended after each GitHub
Actions run.

**Schema:**
```
date, dan_kibble, sanbo_kibble, hand_feeding, compensation, video_count
```

**Weekly digest:** Second GitHub Actions job, runs every Monday at 7:00am
Thailand time (`0 0 * * 1` UTC). Reads last 7 rows of CSV, sends Telegram
summary:
- Dan avg kibble/day
- Sanbo avg kibble/day
- Hand-feeding count this week
- Lowest eating day (flag for health concern)

Same workflow file as daily run, second job definition.

---

## Phase C: Smarter Recording + Data Flywheel
*(Planned after Phase B is stable)*

### C1 — Bowl ROI Zone Filter

Add a configurable region-of-interest mask to `motion_recorder.py`. Motion
outside the bowl area (e.g. room lights changing, shadows) is ignored. Reduces
junk clips and improves cat-filter accuracy.

Config: `BOWL_ROI = (x1, y1, x2, y2)` in `config.py`, default `None` (disabled).

### C2 — Clip Tagging with Cat Identity

After YOLOv8n confirms a cat is present, run a lightweight Dan/Sanbo classifier
on the Pi to tag the clip filename:

```
motion_20260308_062500_30s_dan.mp4
motion_20260308_062800_25s_sanbo.mp4
motion_20260308_063100_45s_unknown.mp4
```

Tags flow through to Drive, making it trivial to find Dan-only or Sanbo-only
clips without running full Colab analysis.

### C3 — Data Flywheel for Future Rover

- Clips where cat identity confidence is low → flagged in Telegram for manual
  review (`/review` command shows flagged clips)
- Manually confirmed clips auto-tagged and added to a `training_candidates/`
  folder on Drive
- Builds labeled dataset passively as the system runs — feeds future rover
  autonomous navigation model

---

## Decisions Made

| Decision | Reasoning |
|----------|-----------|
| GitHub Actions over Kaggle | Same service account cost; runs existing notebook via papermill with zero migration; better debugging UI; cron built-in |
| GitHub Actions over Colab scheduled | Colab has no confirmed free scheduling API |
| One-tap Telegram prompt NOT used | GitHub Actions is clean enough; one-tap was a fallback |
| rclone bisync kept | User intentionally uses two-way sync to clean Pi from Drive side |
| Colab kept for manual use | Transparency for threshold tuning; GitHub Actions handles automation |
| Phase B before C | System is young; need reliability and observability before adding smarter features |

---

## Out of Scope (this phase)

- Rover / movable camera platform (future hardware project)
- Real-time alerts during feeding
- Multi-camera support
- Web dashboard
