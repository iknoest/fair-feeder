# Fair Feeder

**AI-powered cat feeding monitor** that tracks who eats what, counts kibble, detects hand-feeding, and sends daily Telegram reports — all from a $15 camera.

<!-- PHOTO: annotated video frame showing Dan at bowl with bounding boxes -->

---

## Why

I have two cats: **Dan** (picky eater) and **Sanbo** (food thief). Every morning I hand-feed Dan, but Sanbo steals his food the moment I look away. I needed to know: *did Dan actually eat enough today?*

## Repo Guide

If you're new to the project, read the files in this order:

- `motion_recorder.py`, `morning_report.ipynb`, `flagging.py`, `roboflow_upload.py`, `schedule_log.py` — production path (root, hardcoded dependencies)
- `notebooks/fair_feeder_v14.ipynb`, `scripts/train.py`, `scripts/download_dataset.py` — training and dataset work
- `notebooks/smoketest.ipynb`, `notebooks/batch_review.ipynb` — interactive analysis and historical review
- `tests/test_flagging.py`, `tests/test_roboflow_upload.py` — unit tests for production modules

## How It Works

```mermaid
graph LR
    CAM[Tapo C210<br/>IR Camera] -->|RTSP| PI[Raspberry Pi 5<br/>24/7 Recording]
    PI -->|rclone| DRIVE[Google Drive]
    DRIVE --> CI[GitHub Actions<br/>Daily cron + heartbeat]
    CI --> YOLO[YOLOv11s<br/>5-class Detection]
    YOLO --> TG[Telegram Bot<br/>Morning Report]

    style CAM fill:#4a9eff,color:#fff
    style PI fill:#e74c3c,color:#fff
    style DRIVE fill:#f39c12,color:#fff
    style CI fill:#333,color:#fff
    style YOLO fill:#9c27b0,color:#fff
    style TG fill:#0088cc,color:#fff
```

1. **Pi 5** runs 24/7 — detects motion, records clips, uploads to Google Drive
2. **GitHub Actions** picks up clips every morning, runs YOLOv11 inference
3. **FeedingTracker** analyzes detections: who was at the bowl, how long, how many kibble eaten
4. **Telegram bot** sends the report with summary, timeline chart, snapshots, and annotated video

## Detection Classes

| Class | What | Why |
|-------|------|-----|
| **Dan** | Tuxedo cat (dark) | Track feeding time |
| **Sanbo** | Calico cat (orange) | Detect food theft |
| **Dan_hand** | My hand near bowl | Track hand-feeding sessions |
| **Bowl** | Food bowl | Reference point for "at bowl" detection |
| **Kibble** | Individual food pieces | Count consumption |

## Sample Report

```
😸 Dan finished breakfast
2026-03-28
06:20:10-06:22:02 (2m 30s)
Start: ~26 kibble
Dan   [########] 100% (~24)
      bowl 1m 46s; bowl from ~06:20:15
Sanbo [........] 0% (~0)
      bowl 0m 00s; bowl from unknown
Hand: none
Flags: 6 frames -> Roboflow (6 sent); top: blip-kibble 3, low-conf-sanbo 2
```

The first line appears in the Telegram push notification so you can see the verdict without opening the message: `😸 Dan finished breakfast`, `😿 Give Dan ~N kibble`, or `🍽️? Feeding machine not working?`. Timestamps in the report use seconds because a bowl can be emptied within a minute. Scheduler delay is recorded in GitHub summaries/logs, not the Telegram message.

<!-- PHOTO: timeline chart showing kibble count, cat presence over time -->

## Model Performance

Current production baseline is V14. V15 has been trained from 155 manually revised April flagged images and looks better than the fresh V14 smoketest rerun, but V14 and V15 still use different Roboflow validation splits. Use a fixed holdout set before deciding whether to deploy V15.

| Model | Validation context | mAP50 | mAP50-95 | Precision | Recall | Read |
|-------|--------------------|-------|----------|-----------|--------|------|
| V13 | pasted standalone val, 54 images | 0.421 | 0.360 | 0.524 | 0.503 | Weak baseline from the recent comparison run |
| V14 | smoketest rerun, 109 images | 0.690 | 0.564 | 0.768 | 0.743 | Large recovery from V13 |
| V15 | standalone/smoketest-style val, 139 images | 0.741 | 0.594 | 0.815 | 0.780 | Incremental gain over V14 rerun |

Improvement:

- V13 -> V14: mAP50 +0.269, mAP50-95 +0.204, precision +0.244, recall +0.240.
- V14 -> V15: mAP50 +0.051, mAP50-95 +0.030, precision +0.047, recall +0.037.
- V15 improves every class versus the V14 smoketest rerun, especially Dan (+0.080 AP50), Kibble (+0.079), and Dan_hand (+0.061).

## Data Flywheel

The model improves itself through automated feedback:

```mermaid
flowchart LR
    A[Daily report] --> B[Auto-flag<br/>suspicious frames]
    B --> C[Upload to<br/>Roboflow]
    C --> D[Human review<br/>~30 min]
    D --> E[Retrain on<br/>Kaggle]
    E --> F[Deploy]
    F --> A

    style B fill:#ff9800,color:#fff
    style D fill:#e91e63,color:#fff
    style E fill:#9c27b0,color:#fff
```

Auto-flagging catches: single-frame hallucinations, contradicting detections, impossible scenarios (hand without cat), kibble count jumps.

Daily flag counts are included in Telegram and logged to `feeding_log.csv` (`flagged_frames`, Roboflow upload/skipped/failed counts, and top flag tags). The log also records `schedule_time` and `start_time` in Europe/Amsterdam time so GitHub schedule delay can be reviewed without putting delay text in Telegram. At retraining time, the monthly trend shows which failure modes are still common.

Model maintenance handbook: [docs/model-improvement-handbook.md](docs/model-improvement-handbook.md)

## 24/7 Camera Position Alert

The Pi recorder monitors whether the bowl stays framed while running 24/7 using the existing `yolov8n.pt` COCO `bowl` class. Keep `yolov8n.pt` available in the recorder working directory on the Pi so the same lightweight model can handle both cat filtering and bowl-position checks.

Default behavior: check every 30 seconds; send Telegram if the bowl is missing or not fully visible for 10 continuous minutes; cooldown 6 hours. Missing bowl alerts start with `🥣?`; clipped/not-visible camera-position alerts start with `👀?`. A right-side bowl position is OK as long as the model can see the full bowl.

After changing `motion_recorder.py`, deploy it to `/home/pi5/Feeder/fair-feeder/`, compile it on the Pi, restart `cat-monitor.service`, and verify the service is active.

## Morning Report Scheduling

The workflow cron is intentionally early: `0 2 * * *` UTC. GitHub Actions scheduled jobs have shown multi-hour trigger delays, so the job compensates by scheduling early and then waiting until `06:35 Europe/Amsterdam` if GitHub happens to start promptly.

Telegram reports do not include scheduler delay. The GitHub Actions run summary records scheduled time, actual start time, scheduler delay, and runtime; `feeding_log.csv` records Amsterdam `schedule_time` and `start_time` for later analysis.

The CI report filters to the morning feeding window, stitches adjacent clips only when their gap is at most 10 seconds, sends one report for a merged event, and suppresses later empty-food clips once an earlier event finished the bowl.

## Architecture

```mermaid
flowchart TB
    subgraph HW["Hardware"]
        CAM[Tapo C210] --> PI[Raspberry Pi 5]
    end
    subgraph CLOUD["Cloud"]
        CI[GitHub Actions<br/>Cron] --> YOLO[YOLOv11s]
        YOLO --> TRACKER[FeedingTracker]
    end
    subgraph FLYWHEEL["Improvement Loop"]
        FLAG[Auto-Flag] --> RF[Roboflow]
        RF --> KAGGLE[Kaggle Training]
    end
    subgraph OUT["Output"]
        TG[Telegram Bot]
        CSV[feeding_log.csv]
    end

    PI -->|rclone| GDRIVE[Google Drive]
    GDRIVE --> CI
    TRACKER --> TG
    TRACKER --> CSV
    YOLO --> FLAG
    KAGGLE -->|new model| GDRIVE

    style CAM fill:#4a9eff,color:#fff
    style PI fill:#e74c3c,color:#fff
    style GDRIVE fill:#f39c12,color:#fff
    style YOLO fill:#9c27b0,color:#fff
    style TG fill:#0088cc,color:#fff
```

## Hardware

| Component | Cost | Role |
|-----------|------|------|
| Tapo C210 | ~$15 | IR camera, 2K, overhead mount |
| Raspberry Pi 5 | ~$60 | 24/7 motion recording + cat filter |
| Total | **~$75** | |

## Tech Stack

| Layer | Tool |
|-------|------|
| Detection | YOLOv11s (Ultralytics) |
| Training | Google Colab / Kaggle (free T4 GPU) |
| Dataset | Roboflow (ir-kibble) |
| OCR | EasyOCR |
| Motion recording | OpenCV MOG2 |
| Cat filter (Pi) | YOLOv8n (0.10 conf) |
| Secrets | Infisical |
| Notifications | Telegram Bot API |
| Storage | Google Drive (rclone) |
| Automation | GitHub Actions (cron) |

## Project Structure

```
fair-feeder/
├── motion_recorder.py       # Pi 5: 24/7 motion + cat filter  ← stays at root (systemd)
├── morning_report.ipynb     # Daily CI pipeline               ← stays at root (GitHub Actions)
├── flagging.py              # Auto-flag suspicious detections  ← stays at root (imported by CI)
├── roboflow_upload.py       # Upload flagged frames            ← stays at root (imported by CI)
├── schedule_log.py          # Schedule/start log helpers       ← stays at root (imported by CI)
├── config.py                # Camera & detection settings      ← stays at root (imported by Pi)
├── data.yaml                # YOLO dataset config (5 classes)
│
├── notebooks/
│   ├── fair_feeder_v14.ipynb    # Model training (Colab/Kaggle)
│   ├── smoketest.ipynb          # Interactive analysis
│   └── batch_review.ipynb       # Historical reprocessing
│
├── scripts/
│   ├── train.py                 # YOLOv11 training CLI
│   ├── download_dataset.py      # Roboflow dataset downloader
│   ├── polygon_to_bbox.py       # Annotation format converter
│   ├── verify_labels.py         # Label verification grid
│   └── debug_yolo_detection.py  # Detection debugging
│
├── deploy/
│   ├── cat-monitor.service      # systemd service definition
│   └── sync_cleanup.sh          # Cron: purge old local videos
│
├── tests/
│   ├── test_flagging.py
│   ├── test_roboflow_upload.py
│   └── legacy_notebook/
│
└── docs/
    ├── blog/                    # Blog posts (EN + ZH-TW)
    ├── guides/                  # Pi SSH, git guides
    └── plans/                   # Design specs
```

## Setup

See [docs/README_GIT_PULL.md](docs/README_GIT_PULL.md) for credentials setup after cloning.
See [docs/README_RPI_SERVICE.md](docs/README_RPI_SERVICE.md) for Raspberry Pi 5 deployment.

## Blog Post

Read the full story: [How I Built an AI Cat Feeding Monitor](docs/blog/fair-feeder-story.md) (also available in [Traditional Chinese](docs/blog/fair-feeder-story-zh-tw.md))

## License

Private project. Not open-sourced.
