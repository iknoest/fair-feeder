# Fair Feeder — GitHub Copilot Handoff

**Date**: 2026-08-19  
**Status**: Production Workflow Restored, Replay Acceptance Complete, Ready for Fresh Copilot Session

---

## 1. Objective

Fair Feeder is a computer-vision feeding monitor designed for Ava to track the breakfast feeding habits of two cats:
- **Dan**: Needs to eat enough food; has a dedicated feeder monitored by a **Tapo C210 IR camera** (top-down infrared RTSP stream).
- **Sanbo**: Prone to stealing food; has a feeder monitored by a **Logitech C925e USB camera** (top-down RGB stream).

Every morning around **06:18–06:30 Europe/Amsterdam**, the cats eat breakfast. The system records motion clips, synchronizes them to Google Drive via `rclone`, processes the footage in GitHub Actions via scheduled workflows, and sends structured feeding reports, snapshots, and video evidence to Ava's Telegram feeder monitor group.

---

## 2. System Architecture

```
[ Raspberry Pi 5 Runtime ]
  ├─ cat-monitor.service (Tapo C210 IR RTSP -> motion_recorder.py)
  │    └─ Non-blocking async finalizer thread + session continuation inheritance
  ├─ usb-monitor.service (Logitech C925e USB V4L2 -> motion_recorder.py)
  │    └─ Low-light motion clip preservation (mean luminance < 15)
  └─ rclone systemd timers -> Auto-sync MP4 clips to Google Drive

[ Google Drive ]
  ├─ GDRIVE_UPLOAD_FOLDER_ID (Tapo IR clips)
  ├─ GDRIVE_LOGITECH_FOLDER_ID (Logitech RGB clips)
  └─ feeding_log.csv (Cross-camera event record updated via update())

[ GitHub Actions Scheduled Morning Report ]
  ├─ TAPO Matrix Leg:
  │    └─ Runs `morning_report.ipynb` (YOLOv11 kibble counting + OCR ground truth)
  │    └─ Delivers summary text, kibble timeline, snapshots, and <45 MB video to Telegram
  └─ LOGITECH Matrix Leg:
       └─ Runs `scripts/logitech_vlm_shadow.py`
       └─ Groups clips into distinct sessions (>10s gap threshold)
       └─ Applies Gamma 2.5 + CLAHE low-light enhancement to LAB L-channel
       └─ Calls Vertex AI Gemini 2.5 Flash with private Roboflow reference anchors
       └─ Sends text report + RAW vs ENHANCED comparison contact sheet to Telegram
```

---

## 3. Current Authoritative Git State

- **Repository**: `https://github.com/iknoest/fair-feeder` (Public repo)
- **Live Pi Deployed Commit**: `13d4c7c` (`fix(recorder): non-blocking async finalization, session continuation inheritance, and low-light cat detection`)
- **Key Recent Commits**:
  - `13d4c7c`: Resolved TAPO synchronous 222s remux block with background finalizer thread; added continuation session inheritance and low-light motion preservation.
  - `2811a03`: Implemented Logitech >10s clip gap session segmentation.
  - `cca0b65` / `290957a`: Separated GHA matrix job concurrency; added ffmpeg size compression.
  - Current HEAD: Reusable `<45 MB` Telegram video guard, composite RAW vs ENHANCED contact sheets, clean `NO_CLIPS` exit handling, and strict production/replay workflow separation.

---

## 4. Frozen Decisions & Architectural Invariants

1. **TAPO is the Production Ground Truth**: Tapo IR OCR + YOLO kibble tracking remains the authoritative production monitor for Dan's breakfast.
2. **Logitech VLM is Shadow / Non-Authoritative**: Logitech Gemini 2.5 Flash operates in shadow mode for visual identity audit and theft detection; it does NOT count kibble.
3. **Roboflow References are Private**: Downloaded ephemerally into `$RUNNER_TEMP` in CI runners; never committed to git.
4. **Event Segmentation Threshold**: Clips separated by $>10\text{s}$ are segmented into separate feeding sessions. Clips separated by $\le 10\text{s}$ may belong to the same session.
5. **Low-Light Semantics**: Low-light recording rule is **LOW-LIGHT MOTION CLIP PRESERVATION** (preserving ambiguous motion evidence for downstream VLM enhancement/analysis). Darkness itself is **never** evidence of a cat.
6. **Theft Warning Semantics**:
   - `feeder identity` = configured camera location (`TAPO` = Dan feeder, `LOGITECH` = Sanbo feeder).
   - `cat identity` = VLM visual model inference.
   - "Dan at Sanbo feeder" means VLM identified Dan at the Logitech station.
7. **No Model / Provider Changes Without User Gate**: Vertex AI Gemini 2.5 Flash is the established provider. Do not switch models or providers without explicit user instruction.
8. **Telegram Video Upload Limit**: Hard maximum is strictly $< 45\text{ MB}$ (enforced via `scripts/telegram_video_guard.py`) before POST to prevent HTTP 413.

---

## 5. Solved vs Verified Status Matrix

| Component / Capability | Status | Verification Type | Evidence |
| :--- | :--- | :--- | :--- |
| **TAPO Async Remux Finalizer** | **SOLVED** | CODE & REPLAY VERIFIED | Background thread prevents recorder blocking |
| **TAPO 150s Session Rollover** | **SOLVED** | REPLAY VERIFIED | 170.0s video generated & delivered (Msg ID: 1726) |
| **TAPO Natural Record-Until-Leaves**| **OPEN** | **PENDING** | Awaiting real natural morning feeding event |
| **Logitech Event Segmentation** | **SOLVED** | LIVE / REPLAY VERIFIED | 3m35s gap split into 2 independent sessions |
| **Logitech Low-Light Enhancement** | **SOLVED** | CODE & REPLAY VERIFIED | Gamma 2.5 + CLAHE boosts luminance from ~8.7 to ~51.2 |
| **Logitech Vertex Gemini 2.5 Flash** | **SOLVED** | LIVE / REPLAY VERIFIED | Dan identified (conf: 0.85/0.90) against Roboflow refs |
| **Telegram Video Size Guard** | **SOLVED** | CODE & TEST VERIFIED | `telegram_video_guard.py` enforces <45 MB limit |
| **Telegram RAW vs ENHANCED UX** | **SOLVED** | CODE & TEST VERIFIED | Single composite image with labeled banners |
| **Scheduled GHA Workflow Health** | **SOLVED** | CODE VERIFIED | Clean separation between production & replay test |

---

## 6. Current Open Issues

### P0 Issues
1. **Natural >150s TAPO Live Acceptance**:
   - The replay harness proved the recorder software rollover logic works past 150s.
   - Natural live verification is **PENDING** until the next natural feeding event occurs where Dan eats for >2m30s uninterrupted.
2. **Logitech Daily Natural Availability**:
   - With low-light motion preservation deployed (`13d4c7c`), verify tomorrow's morning window records and synchronizes Logitech clips without manual intervention.

---

## 7. Parked Tasks (Do Not Work on Without User Request)

- Cat identity fine-tuning or retraining YOLOv11 V14/V15 models.
- Hardware lighting additions or physical camera relocation.
- Migrating from Vertex AI Gemini 2.5 Flash to other LLM providers.
- Building a new manual Roboflow labeling flywheel.

---

## 8. Known Failure History & Lessons Learned

1. **Synchronous Remux Blocked Recorder**: `ffmpeg` blocked the main event loop for ~222s on the Pi, causing continuation chunks to expire. *Fix: Async background worker thread.*
2. **Low-Light Clip Deletion**: Low-light footage (<15 mean luminance) dropped below YOLOv8n confidence thresholds and was deleted before VLM could evaluate it. *Fix: Low-light motion preservation mode.*
3. **Incorrect Event Merging**: Multiple feeding visits separated by minutes were concatenated into one single session. *Fix: >10s gap segmentation.*
4. **Telegram HTTP 413 "Request Entity Too Large"**: Videos >50 MB failed to upload. *Fix: Deterministic target bitrate calculation & compression guard (`scripts/telegram_video_guard.py`).*
5. **Replay Step Contamination**: Test-only acceptance scripts accidentally triggered during scheduled runs. *Fix: Strict `date_override == 'REPLAY_TEST'` branching; scheduled runs execute only production notebooks and shadow scripts.*
6. **False Infrastructure Failures on 0 Clips**: Days with zero feeding clips failed the entire CI workflow with exit code 1. *Fix: `[NO_CLIPS]` clean exit code 0 with diagnostic logs.*

---

## 9. Desired User-Visible Telegram UX

### TAPO Morning Report (Dan Feeder)
```text
[TAPO] 😸 Dan finished breakfast
2026-08-19
06:20:05-06:22:55 (2m 50s)

Start: ~55 kibble
Dan 100% (~29)
bowl 1m45s; bowl from ~06:20:13
[Attached: Timeline Photo, Clean Dispensed Snapshot, Clean Eating Snapshot, Compressed <45 MB MP4 Video]
```

### Logitech Shadow Report (Sanbo Feeder / Theft Check)
```text
[SHADOW] 2026-08-19 Logitech Feeding Analysis
Recorded Sessions: 1 distinct event(s)

=== EVENT 1 (06:19:55-06:21:53, 1m 58s) ===
😸 Dan feeding session
      cat: Dan ⚠️ Dan at Logitech/Sanbo feeder — verify
      identity basis: body size, contrast-enhanced coat patterns
      visibility: low
      confidence: 0.85
   eating: yes
     bowl: low
     hand: none observed
[Attached: Single composite photo containing RAW (top) vs ENHANCED Gamma 2.5 + CLAHE (bottom)]
```

---

## 10. Test Commands

Run the full verified test suite:
```bash
pytest tests/test_recorder_stabilization.py tests/test_logitech_vlm_shadow.py tests/test_telegram_video_guard.py
```

Compile check:
```bash
python3 -m py_compile scripts/logitech_vlm_shadow.py scripts/telegram_video_guard.py scripts/send_tapo_replay_test.py motion_recorder.py
```

Check git diff for whitespace / privacy leaks:
```bash
git diff --check
```

---

## 11. Pi Deployment Procedure

When modifying Pi runtime files (`motion_recorder.py`):
1. Copy updated script to Pi via SCP.
2. Verify Python syntax on the Pi: `python3 -m py_compile motion_recorder.py`.
3. Restart service: `sudo systemctl restart cat-monitor.service` (or `usb-monitor.service`).
4. Check status: `systemctl is-active cat-monitor.service`.
5. Check logs: `journalctl -u cat-monitor.service -n 20 --no-pager`.
6. **Constraint**: Do NOT trigger unnecessary restarts to avoid spamming Ava with startup notifications.

---

## 12. GitHub Actions Debugging Procedure

1. List recent runs: `gh run list --workflow=morning-report.yml --limit 5`.
2. View job logs: `gh run view --log --job=<JOB_ID>`.
3. View failed step: `gh run view --log-failed --job=<JOB_ID>`.
4. Trigger manual test run:
   - For specific date: `gh workflow run morning-report.yml -f date_override=20260818`
   - For replay acceptance: `gh workflow run morning-report.yml -f date_override=REPLAY_TEST`

---

## 13. Privacy & Public Repository Safety Rules

- **NEVER commit**:
  - Cat or home photos (`.jpg`, `.png`).
  - Video clips (`.mp4`, `.avi`).
  - Credentials, `.env`, tokens, chat IDs, service account keys.
  - Temporary artifacts under `scratch/` or `tmp/`.
- **NEVER use `git add .` or `git add -A`**. Stage exact file paths only.

---

## 14. Exact First Task for GitHub Copilot

**Task**: Inspect the next natural scheduled morning feeding run (target date `2026-08-19` or `2026-08-20`) after the morning feeding window (06:18–06:30 Europe/Amsterdam).

**Execution Steps**:
1. Check the scheduled GitHub Actions workflow run for that morning via `gh run list --workflow=morning-report.yml --limit 3`.
2. Inspect the TAPO job log and delivered Telegram video duration:
   - If duration $>150\text{s}$ and playback shows Dan eating continuously until leaving: Mark **NATURAL $>150\text{s}$ ACCEPTANCE: PASS**.
   - If duration $\le 150\text{s}$ but cat left naturally: Note event duration.
3. Inspect the Logitech job log:
   - Verify if clips were found and processed by Gemini 2.5 Flash.
   - Verify composite RAW vs ENHANCED contact sheet was delivered to Telegram.
4. Report findings concisely to Ava in Traditional Chinese.

---

## 15. Acceptance Criteria

- **TAPO Natural Milestone**: Proven only when a natural live feeding session exceeding 150s records and stitches all chunks until the cat departs without a 2m30 cutoff.
- **Logitech VLM Milestone**: Proven when enhanced low-light footage produces accurate identity inference with RAW vs ENHANCED contact sheet delivered.
- **Production Pipeline**: Must run green every morning without manual intervention or test step contamination.

---

## 16. User Interaction Guidelines for GitHub Copilot

- **Language**: Discuss in **Traditional Chinese (繁體中文)**; technical commands and log extracts in English.
- **Mode**: Autonomous Goal Mode. Execute approved routine fixes without stopping for trivial confirmations.
- **Gates**: Stop only for genuine architectural changes, model/provider migrations, or destructive operations.
- **Integrity**: Never invent or fabricate missing numbers or API outputs. If data is missing, report `NOT MEASURED / UNKNOWN`.
- **Format**: Always conclude responses with a concise summary (Solved, Open, Parked) and a single next actionable step.
