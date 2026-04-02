# CLAUDE.md — Fair Feeder

## 1. PROJECT OVERVIEW

**Fair Feeder** is a computer-vision cat feeding monitor that uses a Tapo IR camera
and YOLOv11 to track two cats (Dan and Sanbo), detect hand-feeding events, count
kibble, and send structured feeding reports via Telegram.

- **Target user:** The project owner — a cat parent who hand-feeds Dan and wants to
  track how much each cat eats, when they arrive, and whether hand-feeding occurred.
- **Core problem:** Without monitoring, there is no way to know if Dan ate his fair
  share or if Sanbo stole food, especially overnight under IR lighting.

---

## 2. PRODUCT REQUIREMENTS (PRD Summary)

### Goals & success metrics
- Correctly attribute kibble eaten per cat (Dan vs Sanbo) from video
- Detect Dan_hand feeding episodes with timestamps
- Send automated Telegram alerts after each video is processed
- Achieve mAP50 ≥ 0.85 on the V13 test split

### Key features (in scope)
- YOLOv11 object detection (5 classes: Dan, Sanbo, Dan_hand, Bowl, Kibble)
- OCR timestamp extraction from Tapo burned-in OSD
- Phase-based eating attribution (proportional to bowl-overlap time)
- Snapshot capture: Sanbo arrival, Dan_hand episodes, kibble-dispensed moments
- Text summary saved to Google Drive (.txt)
- Annotated video output (boxes only, no labels)
- Telegram bot notification (summary + snapshots + video)
- Secret management via Infisical

### Out of scope
- Real-time live alerting (currently batch-processes recorded videos)
- Multi-camera support (single Tapo C210 only)
- Automatic retraining pipeline
- Web dashboard or UI
- Feeding scheduling or automatic dispenser control

### Non-functional requirements
- Runs on Google Colab (free T4 GPU) — no dedicated server needed
- Inference speed: < 50ms per frame at 1280px on T4
- Video files compressed with ffmpeg before Telegram upload; > 50 MB after compression falls back to Drive path
- Secrets must never be hardcoded in committed code (Infisical for API keys)

---

## 3. TECHNICAL ARCHITECTURE

### Tech stack
| Layer | Technology | Why |
|-------|-----------|-----|
| Detection model | YOLOv11s (Ultralytics) | Best accuracy/speed trade-off for 5-class detection |
| Training | Google Colab / Kaggle (T4 GPU) | Free GPU, no local hardware needed |
| Dataset | Roboflow (ir-kibble v13) | Managed labelling + versioning + export |
| OCR | EasyOCR | Reads Tapo's burned-in timestamp from video frames |
| Camera | Tapo C210 (RTSP + ONVIF) | IR night vision, 2K resolution, affordable |
| Motion recording | MOG2 background subtraction | Frame-based, no ONVIF dependency, proven on Pi 5 |
| Live detection | YOLOv8n (0.10 conf) | Local real-time cat detection (`motion_recorder.py`) |
| Cat identification | Custom histogram analysis | Distinguishes Dan (tuxedo/dark) from Sanbo (calico/orange) |
| Secret management | Infisical REST API | Stores Roboflow key, Telegram credentials (ARM64 compatible) |
| Notifications | Telegram Bot API | Sends summaries, photos, video to owner's phone (two-way) |
| Storage | Google Drive (rclone bisync) | Persistent storage for models, videos, outputs |
| Auto-flagging | Custom Python (flagging.py) | Detects suspicious YOLO predictions for human review |
| Data flywheel | Roboflow Upload API | Sends flagged frames to Roboflow for relabeling → retrain |
| Automation | GitHub Actions (cron) | Morning kibble report — runs morning_report.ipynb via papermill |
| Experiment tracking | Weights & Biases | Training metrics, loss curves, checkpoints |

### Project structure
```
fair-feeder/
├── CLAUDE.md                  # This file
├── config.py                  # Camera, detection, identification settings
├── train.py                   # YOLOv11 training CLI
├── download_dataset.py        # Roboflow dataset downloader
├── polygon_to_bbox.py         # Convert polygon annotations → YOLO bbox
├── verify_labels.py           # Visual label verification grid
├── tapo_check.py              # RTSP connection tester
├── motion_recorder.py         # 24/7 Motion-triggered recording + YOLO cat filter
├── README_GIT_PULL.md         # Setup guide for credentials after git pull
├── README_RPI_SERVICE.md      # Pi 5 deployment guide
├── test_env.py                # Environment validation
├── data.yaml                  # YOLO dataset config (5 classes)
├── requirements.txt           # Core dependencies
├── fair_feeder_v13.ipynb      # Training notebook (Colab/Kaggle)
├── smoketest.ipynb            # Inference + feeding analysis (staged pipeline)
├── morning_report.ipynb       # CI pipeline notebook (papermill, GitHub Actions)
├── batch_review.ipynb         # Colab notebook for historical video reprocessing
├── flagging.py                # Auto-flag suspicious YOLO detections
├── roboflow_upload.py         # Upload flagged frames to Roboflow
├── test_flagging.py           # Unit tests for flagging module
├── test_roboflow_upload.py    # Unit tests for upload module
├── test_yolo_detection.py     # Utility to debug YOLO on Pi camera
├── test_notebook_fixes.py     # Unit tests for notebook patch helpers
├── sync_cleanup.sh            # Auto-purge cron job to delete old local videos
└── tasks/
    ├── todo.md                # Current task tracking (checkable items)
    └── lessons.md             # Self-improvement log (updated after corrections)
```

### Key dependencies
- `ultralytics` — YOLOv11 training & inference
- `roboflow` — dataset download + flagged frame upload (data flywheel)
- `easyocr` — timestamp OCR
- `opencv-python` — video/image processing, RTSP frame reading, recording
- `mediapipe` — real-time detection (main.py, motion_recorder.py cat filter)
- `onvif-zeep-async` — ONVIF camera event subscription (motion detection)
- `infisicalsdk` — secret management (pip package renamed from `infisical-sdk`; import remains `from infisical_sdk`)
- `requests` — Telegram Bot API calls

### External services
| Service | Purpose | Auth method |
|---------|---------|-------------|
| Roboflow | Dataset (ir-kibble v13) | API key via Infisical |
| Infisical | Secret vault | `INFISICAL_ID` / `INFISICAL_SECRET` / `INFISICAL_PROJECT_ID` in Colab Secrets |
| Telegram Bot | Notifications | `TelegramBotToken` + `TelegramChatId` via Infisical |
| Google Drive | File storage | Colab `drive.mount()` |
| Tapo C210 | RTSP video source | Username/password in env vars |
| Weights & Biases | Training metrics | API key in Colab Secrets |

---

## 4. CURRENT PROJECT STATUS

**Stage: Motion detection working on Pi 5; cat detection needs AI model fix**

### Completed on Raspberry Pi 5
- [x] **RTSP camera connection** - Tapo C210 connects via TCP transport (UDP too unreliable)
- [x] **Motion detection (MOG2)** - Frame-based background subtraction working in `test_motion_pi.py`
- [x] **Video recording** - Captures motion videos with 3s pre-buffer
- [x] **Google Drive sync** - `rclone bisync` uploads videos in background (fire-and-forget)
- [x] **Automatic storage cleanup** - Deletes no-cat videos to save space
- [x] **24/7 monitoring ready** - Can run via systemd service
- [x] **TCP RTSP transport fix** - Solved network unreliability on Pi 5
- [x] **Credentials from config.py** - No hardcoded secrets in code

### Working for Production (Raspberry Pi 5)
- [x] **Cat detection filter** - `motion_recorder.py` uses YOLOv8n (at 0.10 threshold) to filter out non-cat videos.
- [x] Efficient CPU inference (YOLOv8n runs at ~300ms/frame on Pi 5)

### Working for Production (Colab Only)
- [x] YOLOv11 V14 model trained (5 classes, mAP50=0.957 overall, Sanbo AP50=0.985)
- [x] Smoketest notebook: full video analysis pipeline (requires GPU, runs on Colab)
- [x] FeedingTracker: phase-based kibble attribution
- [x] Telegram bot integration with video uploads
- [x] All AI analysis features (timestamp OCR, kibble counting, hand-feeding detection)

### Working for Production (GitHub Actions / CI)
- [x] **Morning kibble report** — `morning_report.ipynb` runs via GitHub Actions at 06:35 Europe/Amsterdam local time
- [x] **CI-compatible notebook** — `RUNNING_IN_CI` guard on all Colab-only cells; `tqdm.auto` replaces `tqdm.notebook`
- [x] **Drive service account integration** — service account auth and file update pattern in place
- [x] **Merged clip detection** — Phase 1 YOLO inference correctly runs on stitched merged video; timeline chart generated correctly
- [x] **FeedingTracker analytics** — Fixed: fallback to phase-entry/exit kibble counts (issue #31)
- [x] **Correct annotated video sent to Telegram** — Fixed: guarded SOURCE_DIR rescan in CI (issue #32)
- [x] **feeding_log.csv accumulating** — Fixed: CSV cell moved after Phase 3, zero-row guard added (issue #34)
- [x] **Duplicate Telegram sends** — Fixed: removed Monday cron + weekly digest job, guarded retry cell
- [x] **Drive video upload** — SA has zero quota; dropped from CI by design. Colab handles archive (issue #33)

### In progress
- [ ] **Phase C: Data Flywheel** — see `docs/superpowers/specs/2026-03-26-data-flywheel-design.md`
  - [x] C1: Auto-flag suspicious detections + upload to Roboflow — verified in CI 2026-03-26
  - [x] C2: Batch reprocessing notebook — 231 flagged frames uploaded to Roboflow with pre-annotations
  - [x] C3: First retrain cycle (v14) — 775 images labeled, v14 trained 2026-03-28
  - [ ] C3: Deploy v14 to CI and verify real-world flag count improvement
  - [ ] C3: Update `GDRIVE_MODEL_FILE_ID` GitHub secret for v14 model

### V14 model improvements (vs V13, validated on V13 test split)
- **Sanbo**: AP50 0.959→0.985, recall 0.881→1.000 — blip-sanbo should drop significantly
- **Dan_hand**: AP50 0.928→0.936, precision→1.000 but recall 0.844→0.716 — fewer false positives, may miss some real events
- **Dan**: AP50 0.974→0.936 — precision dropped 0.948→0.867, needs monitoring in real-world runs
- **Kibble**: AP50 0.924→0.931 — slight improvement despite removing copy_paste augmentation
- **Overall**: mAP50 0.956→0.957, mAP50-95 0.739→0.754 (tighter boxes)

### Planned later
- [ ] Bowl ROI zone filter in `motion_recorder.py`
- [ ] Lightweight Dan/Sanbo classifier on Pi — tag clip filenames
- [ ] Telegram-interactive flagging (reply to flag issues)

---

## 5. RASPBERRY PI 5 DEPLOYMENT STATUS

### Hardware Specifications (Pi 5 Setup)
```
Device:           Raspberry Pi 5
CPU:              64-bit Quad-core ARM Cortex-A76
RAM:              4GB / 8GB (user's configuration)
Storage:          microSD card (recommend 32GB+ for video buffer)
Network:          WiFi 6 802.11ax (dual band)
Camera Input:     Tapo C210 (2304×1296 via RTSP)
Output:           Google Drive via rclone
```

### What Works on Pi 5 ✅
- Motion detection (MOG2 background subtraction) - ~10% CPU
- RTSP video streaming - stable with TCP transport
- Video encoding (mp4v codec) - real-time at 15 fps
- Google Drive sync (rclone) - background process
- 24/7 monitoring - systemd service capable

### Limitations & Known Issues ⚠️

#### 1. **AI Model Performance on Extreme Angles**
- **Problem**: EfficientDet models hallucinated bounding boxes ("oven", "sink") when the camera was placed at ground level aimed at the cat bowl.
- **Solution**: Switched to YOLOv8n in `motion_recorder.py` and dropped the confidence threshold to `0.10`. YOLOv8n correctly identifies partial cat bodies in these extreme edge cases.

#### 2. **Storage Constraints**
- **Problem**: Continuous video recording fills storage rapidly
  - 2304×1296 @ 15fps ≈ 50-100 MB per minute uncompressed
  - Tapo IR + motion blur = poor ffmpeg compression ratio
- **Solution**: 
  - Save only when cat detected (cat filter working = ~10% of motion events are cats)
  - Use `rclone copy` to upload instantly without `.lck` deadlocks
  - Use a daily cron job (`sync_cleanup.sh`) to automatically purge local files older than 3 days
  - You can check storage and sync health via the Telegram `/syncstatus` command

#### 3. **Python Environment Fragmentation (ARM64 vs x86)**
- **Problem**: Pre-built Python dependencies (wheels) drop support or break APIs on the Raspberry Pi's ARM architecture.
  - `ai-edge-litert` is installed natively on the Pi OS as the modern replacement for `tflite-runtime`, but its API can differ or crash unexpectedly compared to older Windows versions.
  - `infisical-sdk` (written in Rust) lacks official pre-built ARM64 binaries and fails to install on the Pi.
- **Solution**: 
  - ALWAYS run `pip list` or query the Pi's native environment to see what is *actually* installed rather than assuming packages are missing.
  - When an SDK fails to build on ARM, pivot to calling its REST API natively using Python `requests` (e.g., Infisical Universal Auth).
  - Use lightweight alternatives when complex packages fail.

#### 4. **Scope Limitations (Unchanged)**
- Only two cats supported (Dan and Sanbo) — class IDs are hardcoded
- Only one camera angle (fixed overhead Tapo C210)
- Batch processing only (no real-time feeding alerts from smoketest pipeline)
- YOLOv11 analysis requires Colab GPU (not feasible on Pi 5 CPU)
- No web UI — all interaction is via terminal + Telegram (if set up)

### Migration Challenges: Development (Windows) → Raspberry Pi 5

| Challenge | Symptom | Solution |
|-----------|---------|----------|
| **RTSP connection fails** | "No route to host" | Use TCP transport, not UDP |
| **Missing TFLite packages** | "No module named 'tensorflow'" | Use ai-edge-litert (lighter weight) |
| **ai-edge-litert API crash** | Noticed API mismatch, assumed edge AI was dead | Verify environment directly; YOLOv8n proved more stable on CPU |
| **Virtual environment bloat** | pip install hangs on ARM wheels | Pre-filtered requirements.txt for Pi |
| **Path separators (W vs U)** | `H:\` Drive letter doesn't exist | Use `platform.system()` checks |
| **File encoding (UTF-16 logs)** | PowerShell output incompatible | Use Python `open()` with utf-8 explicit |
| **Google Drive not mounted** | "Permission denied" on /gdrive | Use rclone instead of Drive API |
| **Infisical SDK fails on ARM** | pip cannot find matching wheel `infisical-sdk` | Use standard `requests` to call Universal Auth REST API directly |
| **Credentials in env vars** | Real password leaked in .py files | Load via REST API / `.env` file |

### Current Recommendations for Pi 5 Deployment

**Use `motion_recorder.py` (Proven working)**
- ✅ Motion detection working (MOG2)
- ✅ Video recording working (3s pre-buffer)
- ✅ Google Drive upload working (`rclone`)
- ✅ Cat filtering enabled (YOLOv8n)
- **Best for**: 24/7 autonomous monitoring with storage optimization (deletes non-cat motion automatically).

### Realistic Expectations: Pi 5 vs Colab

| Feature | Pi 5 | Google Colab |
|---------|------|---------------|
| 24/7 motion monitoring | ✅ Yes | ❌ No (session limit) |
| Motion detection accuracy | ✅ High (MOG2) | ✅ High |
| Cat detection (TFLite) | ⚠️ Unstable | ✅ High (GPU-backed) |
| YOLOv11 video analysis | ❌ No (CPU bound) | ✅ Yes (T4 GPU) |
| Real-time inference | ❌ Too slow | ✅ <50ms/frame |
| Storage management | ⚠️ Manual cleanup | ✅ Automatic (Drive) |
| Telegram integration | ✅ Yes | ✅ Yes |
| Cost | ✅ ~$60 one-time | ✅ Free (limited) |

### Recommended Architecture for Production

```
Raspberry Pi 5 (24/7 Motion Recorder)
├─ test_motion_pi.py OR motion_recorder.py
├─ Detects motion + records video
├─ Uploads to Google Drive instantly (rclone copy)
└─ sync_cleanup.sh (cron) deletes local videos > 3 days old
        ↓
Google Drive (Video Storage)
├─ Stores raw videos from Pi
└─ Mounted as H:\ on Colab
        ↓
Google Colab (Daily Batch Analysis)
├─ Downloads videos from Drive
├─ Runs YOLOv11 analysis
├─ Generates feeding summaries
└─ Sends Telegram report
```

---

## 6. ISSUES LOG

### Resolved issues

| # | Issue | Root cause | Fix | Commit |
|---|-------|-----------|-----|--------|
| 1 | OCR timestamps had spaces between every character: `2 0 2 6 - 0 1` | easyOCR reads chars individually | Strip all spaces, regex captures date+time groups, rejoin with single space | 6bc37dc |
| 2 | OCR timestamps missing space between date and time: `2026-01-2509:51:5` | Regex required exactly 2 digits for seconds; partial reads fell through to raw text | Changed regex to `\d{1,2}` for seconds, zero-pad partial components | c4bbf85 |
| 3 | Dan_hand snapshot saved but summary says 0 attempts | `process_frame()` saved snapshots eagerly; `summarize()` filtered short episodes, leaving orphans | Orphan snapshot cleanup after episode filtering | 6bc37dc |
| 4 | Dan_hand false positives without Dan body present | No co-detection requirement | Added `dan_here` check: Dan_hand requires Dan body in same frame | 6bc37dc |
| 5 | Kibble count flickers 0→1→0→2→1 per frame | Same kibbles detected/undetected as they move | Rolling median smoothing (window=3) | 6bc37dc |
| 6 | Eating attribution double-counted kibble across overlapping phases | Per-episode accounting instead of phase-based | Rewrote with phase-based attribution + double-counting guard | 988a8ca |
| 7 | Video cell failed when run independently (missing `video_paths` var) | Cell depended on prior image cell's variable | Re-scan `SOURCE_DIR` at top of video cell | 6f4c64e |
| 8 | `model.val()` gave wrong metrics (0.000 for some classes) | Roboflow exported polygon annotations; YOLO dropped them during val | Added polygon→bbox conversion before validation | bfaeddc |
| 9 | Per-class AP50 showed wrong class names | YOLO sorts classes alphabetically; index 0 ≠ class 0 | Used `model.names` dict for correct index mapping | fc9078f |
| 10 | `imgsz=1280` warning about stride-32 | Passing tuple instead of int | Changed to single int `imgsz=1280` | f8ede5d |
| 11 | Telegram message too wide — long `━` separators stretched bubble | Fixed-width characters forced full-width bubble on mobile | Replaced with short `── Section ──` style headers | f9fe9d5+ |
| 12 | Timestamps in summary showed full date+time for every event | No date deduplication | `_fmt_time()`: strip date if same as video start date | f9fe9d5+ |
| 13 | Video > 50 MB silently fell back to Drive path with no inline playback | Bot API limit; no compression step | Added ffmpeg H.264 compression (crf=28, 720p) before upload | f9fe9d5+ |
| 14 | All videos sent to Telegram only after all processing finished | `video_summaries` collected first, then sent in cell 15 | Moved send call into cell 14's per-video loop | f9fe9d5+ |
| 15 | Tapo credentials hardcoded in config.py | Fallback default values in source | `config.py` now loads from Infisical; falls back to env vars | f9fe9d5 |
| 16 | `motion_recorder.py` TypeError on `create_pullpoint_manager` | Missing `subscription_lost_callback` keyword arg | Added the callback parameter | — |
| 17 | Recording stops and restarts during continuous motion | Tapo ONVIF firmware sends events in bursts with 1-3s gaps (debounce) | Changed stop logic to use `last_motion_time` timer instead of instantaneous flag; only stops after full 5s with no event | — |
| 18 | Tapo password hardcoded in `motion_recorder.py` fallback | Default value contained real password | Replaced with `<YOUR_CAMERA_PASSWORD>` placeholder; credentials via env vars | — |
| 19 | GitHub Actions workflow silently received empty secrets | Secrets added to repo settings but not listed in the step's `env:` block; `os.environ.get()` returned `''` | Listed every required secret explicitly under `env:` in the papermill step | f804329 |
| 20 | `tqdm.notebook` ImportError in CI (IProgress not found) | `tqdm.notebook` requires `ipywidgets.IntProgress`; no widget server in papermill/CI | Replaced all `tqdm.notebook` imports with `tqdm.auto` | f804329 |
| 21 | `feeding_log.csv` create raised `HttpError 403 storageQuotaExceeded` on first CI run | Service accounts have zero storage quota on personal Google Drive; `files().create()` uses SA quota | Pre-created file in Drive UI, shared with SA (Editor); wrapped `create()` in try/except; CI only calls `update()` | a0e1853 |
| 22 | CI ran stale notebook code despite fix being pushed | Workflow triggered just before push landed; GitHub checked out pre-fix commit | Re-triggered workflow manually after confirming fix was on main | — |
| 23 | `motion_recorder.py` videos played back ~1.7× sped up | `cv2.VideoWriter` hardcoded at `VIDEO_FPS=15`; RTSP stream delivered fewer real frames/s under load; each frame stamped `1/15 s` apart | Read `cap.get(cv2.CAP_PROP_FPS)` after connect and use as writer FPS; count actual frames written; on stop, remux with ffmpeg `setpts` if actual vs declared rate diverges >20% | — |
| 24 | CI workflow crashed with `FileNotFoundError: ffmpeg` on first Telegram compression | ffmpeg not installed in GitHub Actions runner | Added `sudo apt-get install -y ffmpeg` step before pip install in `morning-report.yml` | — |
| 25 | Morning report processed every video in Drive on every run | No date/time filter applied to Drive file list | Added `_in_feeding_window()` filter in `smoketest.ipynb` to match filenames against 06:18–06:30 window; multiple clips stitched with ffmpeg concat before analysis | — |
| 26 | `smoketest.ipynb` falsely truncates videos to 2 seconds | The "Empty Bowl Early Exit" optimization exited immediately if no cats were seen | Disabled empty bowl exit threshold by setting to 9999 for full video rendering | — |
| 27 | `feeding_log.csv` stopped accumulating data (only had 1 day visible) | The CI runner creates a fresh 1-row CSV each run and blindly called `update()` on GDrive, overwriting the entire historical file | Always explicitly download the existing CSV from Google Drive (`get_media()`) to memory before appending local rows, THEN push the updated file | — |
| 28 | `IndentationError: unexpected indent` when injecting Python into Jupyter Notebook API | The injected Python multi-line string was aligned at 8-spaces, but the surrounding block in the Notebook JSON arrays was at 4-spaces | Verify the explicit local block indentation level of surrounding syntax before modifying `cell['source']` natively with string replacement scripts | — |
| 29 | `IndentationError` or magical parsing failures in Papermill | Python `write_to_file` on Windows generates `\r\n` carriage returns. Passing these into Jupyter JSON breaks the IPython lexer | Always explicitly `.replace('\r', '')` when pushing string lists into Jupyter Notebook `source` structures on Windows | — |
| 30 | Video files requested for `Today` missed morning captures | Datetimes default to UTC. If a file is requested at 07:00 CET, UTC yields the previous day (`23:00 UTC`) | Always strictly establish the target timezone (e.g. `pytz.timezone('Europe/Amsterdam')`) when conducting string datetime queries against APIs | — |
| 31 | FeedingTracker reports "Start: ~0 kibble / No activity" despite detection timeline showing kibble 0–12, Dan at bowl, and Sanbo at bowl throughout the merged video | `_find_clear_kibble_count` searches no-cat frames; model only detects kibble when cats are present, so "clear" frames always return 0 | Added `_find_kibble_at_phase_entry` / `_find_kibble_at_phase_exit` fallback methods to FeedingTracker | 826bc52 |
| 32 | Telegram sends unmerged short clip annotated video (`_25s_annotated.mp4`) instead of merged annotated video | Phase 1/2 re-scanned SOURCE_DIR, overwriting `video_paths` set by the stitch cell; `merged_names` used wrong variable | Guarded re-scan behind `if not RUNNING_IN_CI:` in Cells 12/13; `merged_names` reads from `merged_sources` dict | ae49d5c+ |
| 33 | Annotated output video never appears in Google Drive output folder after CI run | SA has zero storage quota on personal Drive; `files().create()` fails with 403 | Decision: drop Drive uploads from CI entirely; use Colab (user account, no SA issue) for archiving. See data flywheel design. | — |
| 34 | `feeding_log.csv` still not accumulating data after `98c3a47` fix | CSV cell ran before Phase 1–3 (summary undefined) + wrong video path | CSV cell moved to after Phase 3; reads from `video_results[-1]['summary']`; skips entirely when no videos processed | ee5c70e |

| 35 | OCR timestamp shows `6:20|:0` — pipe character in timestamp | EasyOCR misreads `:1` as `|:` from Tapo's thin OSD font | Replace `|:` with `:1` then remaining `|` with `1` before regex parsing; order matters | 34f7e86 |
| 36 | Kibble Y-axis on timeline chart hard to read at high counts | No explicit Y-axis limit; matplotlib auto-scaled without padding | Added `set_ylim(0, max_k * 1.15)` with `max N` text annotation | e5ea2ae |
| 37 | Dan ate ~40 kibble but chart shows max 26 visible | Double-counting guard skipped when video ends mid-feeding (`last_clear=None`); also `first_clear` underestimated starting kibble due to cat occlusion | Guard now uses `peak_kibble = max(kibble_counts)` as starting estimate; added `last_clear` fallback via `_find_kibble_at_phase_exit` | 3a6ba77 |

### Unresolved
- **Drive video upload from CI (issue #33)** — dropped by design; Colab handles archive instead

---

## 7. DECISIONS MADE

| Decision | Reasoning | Alternatives rejected |
|----------|-----------|----------------------|
| YOLOv11s (small) over YOLOv11m/l | Best speed/accuracy for Colab T4; small objects (kibble) benefit from 1280px input more than model size | YOLOv11m (slower, marginal gain); YOLOv11n (too inaccurate for kibble) |
| 1280px inference size | Kibble is tiny; high resolution critical for detection | 640px (default — missed too many kibbles) |
| Phase-based eating attribution | Handles overlapping feeding (both cats at bowl simultaneously) correctly | Per-episode counting (double-counted shared phases) |
| EasyOCR for timestamps | Works on Tapo's burned-in OSD; no camera API needed | pytesseract (worse on thin OSD fonts); Tapo API (no timestamp endpoint) |
| Boxes-only video (no labels) | Labels + percentages cluttered the view and obscured kibble | Show labels (too noisy); separate labeled/unlabeled videos (doubles file size) |
| Telegram over Discord | Owner uses Telegram; bot API supports photos + video natively | Discord webhooks (limited file handling; owner doesn't use Discord daily) |
| Infisical for secrets | Centralized secret management; works with Colab Secrets for auth | Colab Secrets only (can't share across notebooks/sessions easily); .env file (gets committed by accident) |
| Rolling median (window=3) for kibble smoothing | Simple, effective, preserves real changes while removing single-frame flicker | Kalman filter (overkill); larger window (loses real transitions) |
| Dan_hand requires Dan body co-detection | Hand can't exist without the cat; eliminates stray false positives | Confidence-only threshold (too many FPs); larger bounding box check (misses edge cases) |
| Copy-paste augmentation at 30% | Fixes kibble class imbalance without distorting real distribution | Oversampling (less diverse); higher rate (introduced artifacts) |
| `rect=True` for training and inference | Preserves 16:9 aspect ratio of Tapo footage; prevents letterbox distortion | Square padding (distorts cat proportions; worse mAP) |
| Early exit on empty bowl (no cats + kibble=0 for 5s) | Long videos with inactive periods waste GPU and analysis time | Never exit early (wastes time); exit on no-cat-detected (too aggressive) |
| ffmpeg compression (crf=28, 720p) for Telegram | Most feeding videos fit under 50 MB after compression; inline playback maintained | Send as document (no compression but no inline playback); skip large videos |
| Per-video Telegram send inside processing loop | User gets result for each video immediately; no waiting for all videos to finish | Batch send at end (user waits longer; retry harder) |
| `_fmt_time()` strips date when same as video start | Redundant dates clutter mobile Telegram bubble; date is already in header | Always show full timestamp (repetitive); never show date (breaks overnight reads) |
| Compensation = `sanbo_kibble_eaten` | Most actionable metric for owner — directly answers "how much extra does Dan need?" | Show percentage only (less actionable) |
| ONVIF events for motion over frame-based MOG2 | Offloads processing to camera; no CPU-heavy background subtraction | MOG2 (high CPU on 2K stream); YOLO-based motion (overkill for trigger) |
| Cat detection filter using EfficientDet (2s interval) | Lightweight, already available locally; avoids saving useless clips (e.g. human walking by) | No filter (fills Drive with junk); YOLO (too heavy for continuous sampling) |
| Delete all no-cat videos regardless of duration | User wants only cat clips saved; short clips are often false triggers | Keep short clips as safety buffer (user rejected) |
| `last_motion_time` timer for recording stop | Tapo sends events in bursts with gaps; instantaneous flag causes premature stops | Per-poll `motion_detected` flag (broken by Tapo firmware debounce) |
| Credentials via env vars with placeholders in source | Git-safe; easy local setup via `$env:TAPO_PASS` | `.env` file (risk of commit); Infisical-only (not available locally) |
| JPEG-compressed frames in detection cache | ~50KB/frame vs ~9MB raw; enables Phase 2 replay without video I/O | Raw numpy arrays (too large, ~9MB/frame); no frames in cache (requires slow video seeking) |
| 3-stage pipeline (cache → analytics → output) | YOLO runs once; analytics re-runnable in <2s for threshold tuning | Monolithic cell (re-runs everything); 2-stage with video seeking (still slow for snapshots) |
| Background rclone sync per recording | Fire-and-forget `subprocess.Popen` keeps Python responsive so it doesn't miss the next motion capture | Synchronous Python upload (blocks the loop and drops frames while uploading) |
| OS platform detection for output paths | Single script runs on both Windows dev environment and Pi seamlessly | Maintaining separate branch or script for Raspberry Pi |
| `tqdm.auto` over `tqdm.notebook` in all notebooks | `tqdm.notebook` requires ipywidgets; crashes in CI with no widget server. `tqdm.auto` works in both Colab and CI | `tqdm.notebook` (CI incompatible); conditional import (fragile) |
| `RUNNING_IN_CI` guard for all Colab-only cells | `google.colab`, `infisical_sdk`, and `drive.mount()` all crash in CI; single env-check flag is cleanest isolation | Try/except per-import (hides errors); duplicate notebook (maintenance burden) |
| Service account uses `update()` not `create()` for Drive files | SA has zero storage quota on personal Drive; `create()` fails with 403. `update()` uses file owner's quota | Granting SA more permissions (unnecessary); skipping logging (loses data) |
| Stitch clips only if gap ≤ 10 seconds | Pi motion detection sometimes splits one continuous feeding event into multiple clips with tiny gaps; clips further apart are genuinely separate events and must not be merged | Stitch all clips in the feeding window regardless of gap (wrong — merges unrelated events and breaks FeedingTracker attribution) |
| Clips with gap > 10s are separate feeding events | Each event gets its own FeedingTracker analysis, its own kibble count, its own verdict, and its own Telegram block — not combined | One merged report per morning (loses per-event detail; kibble attribution becomes meaningless across a large time gap) |
| Weekly digest dropped — no scope | Daily Telegram report already answers the key question (did Dan eat?); weekly summary adds no actionable information on top of that | Weekly digest (B4) — removed from roadmap |
| CI = morning cron only; manual analysis = Colab smoketest notebook | CI automation handles the daily scheduled run; interactive threshold tuning and ad-hoc checks are done by running smoketest.ipynb manually in Colab | Using CI for manual analysis runs (unnecessary complexity; Colab is faster for interactive use) |
| Drop Drive uploads from CI; use Colab for archive | SA has zero quota on personal Drive; `files().create()` always fails with 403. Colab mounts Drive as the user — no quota issue. Telegram already delivers daily results. | User OAuth2 token in CI (extra setup, token management); pre-create placeholder files (impractical for daily new files); rclone in CI (same token issue, more moving parts) |
| Auto-flag suspicious detections + upload to Roboflow | Closes the model improvement loop: low-confidence, single-frame blips, and conflicting detections auto-extracted and sent to Roboflow for relabeling. No manual frame extraction needed. | Manual frame extraction (tedious, error-prone); Drive staging folder (extra review step before Roboflow); Telegram-interactive flagging (more complex, can layer later) |
| Roboflow SDK for uploads over raw REST API | `roboflow` package already in dependencies; SDK handles retries, auth, tag lists cleanly | Raw `requests.post()` (more code, manual error handling, tag format differences) |
| Batch by month in Roboflow (`flagged-YYYY-MM`) | Keeps batch count manageable (~12/year) while images appear immediately for review | Daily batches (180/year clutter); no batches (hard to filter by time period) |
| Separate `batch_review.ipynb` for historical reprocessing | Keeps `smoketest.ipynb` focused on daily CI. Batch notebook has different concerns (no feeding window filter, Drive output, multi-video summary). | Overloading smoketest.ipynb with more modes (complexity); ad-hoc Colab cells (not reproducible) |
| V14: Disable copy_paste augmentation | Kibble is already the largest class (4015 annotations). copy_paste=0.3 amplified the wrong class and risked hallucinated kibble. Data bottleneck is Sanbo (293) and Dan (438), not Kibble. | Keep copy_paste=0.3 (was helpful in v13 but v14 data balance is different) |
| V14: Keep YOLOv11s over larger models | 775 training images is too few for YOLOv11m (20M params) — overfitting risk. Small object detection comes from 1280px resolution, not model size. Rule of thumb: ~1000-2000 images per million params. | YOLOv11m (more capacity but overfits on small dataset); YOLOv11l (even worse data-to-param ratio) |
| Upload pre-annotations to Roboflow with `is_prediction=True` | Reviewer sees what the model predicted (boxes) and corrects them, rather than labeling from scratch. Faster review, fewer missed corrections. | Upload images only (reviewer labels from scratch — slower, may miss subtle errors) |
| Track uploaded frames in `roboflow_uploaded.txt` on Drive | Prevents duplicate uploads across batch_review sessions. Simple append-only text file, one frame ID per line. | Roboflow API dedup (limited, based on image hash not frame ID); no tracking (risk of flooding Roboflow with duplicates) |
| pip package `infisicalsdk` (not `infisical-sdk`) | Package renamed upstream; `infisical-sdk` is deprecated. Import name stays `from infisical_sdk import ...` | Keep old package name (will stop receiving updates) |
| `peak_kibble` (max visible) for double-counting guard, not `first_clear` | `first_clear` measures kibble when cat just arrived — body already occludes some. `max(kibble_counts)` is the best camera angle with most kibble simultaneously visible, closest to true bowl count | `first_clear` (underestimates by 30-40% due to occlusion); phase_entry median (still measured with cat present) |
| OCR: replace `\|:` with `:1` before `\|` with `1` | Tapo OSD font causes EasyOCR to read `:1` as `\|:`. Replacing `\|` alone gives `6:201:0` (extra colon). Must replace `\|:` → `:1` first | Single `\|` → `1` replacement (wrong when followed by colon) |

---

## 8. PENDING / NEEDS CLARIFICATION

### Resolved ✓
- **Infisical secret names for Telegram** — `TelegramBotToken` and `TelegramChatId` confirmed.
- **Tapo camera credentials** — `TAPO_IP`, `TAPO_USER`, `TAPO_PASS` moved to Infisical with
  fallback to env vars for local use (main.py). Updated `config.py` to load from Infisical
  when available.
- **Automated scheduling** — Owner wants automatic video processing pipeline; smoketest runs
  manually in notebook. Next phase: schedule Drive uploads or integrate trigger system.
- **Additional cats** — Not expected; no need for multi-cat architecture redesign.
- **Model versioning** — Will use lightweight `MODELS.md` file (git-tracked) to log each
  trained model: name, mAP50, date, Colab commit, Drive path, notes.

- **Automated scheduling** — Resolved: morning kibble report runs via GitHub Actions with timezone-aware schedule (`35 6 * * *`, `Europe/Amsterdam`) for 06:35 local time year-round.

### Still open (nice-to-have)
- ~~Weekly digest (B4)~~ — dropped; no value on top of daily report

---

## 9. WORKFLOW ORCHESTRATION

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, **STOP and re-plan immediately** — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review `tasks/lessons.md` at session start

### 3. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 4. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 5. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user

---

## 10. TASK MANAGEMENT

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

### When to update each file

| File | Update when | What goes in it |
|------|------------|----------------|
| `CLAUDE.md` | Phase status changes, new issue resolved, new architectural decision, new coding convention established | Permanent project knowledge — any new agent session reads this cold |
| `tasks/lessons.md` | A mistake was corrected or non-obvious behavior was discovered through debugging | Anti-patterns + how to avoid them; numbered rows, consistent style |
| `tasks/todo.md` | Task state changes — items completed, added, or started | Current work items only; completed items move to Archived section |

**Rule of thumb:** CLAUDE.md = *what is true about this project*. lessons.md = *what went wrong and how to avoid it*. todo.md = *what needs to be done*.

A resolved issue can appear in both CLAUDE.md's Issues Log (what was fixed + commit) and lessons.md (the generalised rule to prevent recurrence) — they are complementary, not redundant.

---

## 11. CORE PRINCIPLES

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

---

## 12. CODING CONVENTIONS

### Style
- Python, no type annotations unless already present in the file being edited
- Use existing helper functions (`draw_boxes`, `bbox_iou`, `parse_results`, etc.)
  before creating new ones
- Notebook cells should be self-contained where possible (re-scan SOURCE_DIR, etc.)
- Class colours defined in `CLASS_COLORS_RGB` dict — always use `get_color_bgr()`
- Detection thresholds live in the config cell (cell 6 in smoketest.ipynb) — do not
  scatter magic numbers

### Always do
- Read the file before editing — understand existing code first
- Preserve the 16:9 aspect ratio (`rect=True`) in all YOLO calls
- Use `model.names` dict for class index mapping (YOLO sorts alphabetically)
- Keep video output as boxes-only (`show_label=False`) — owner explicitly requested this
- Save text summaries to Google Drive alongside other outputs
- Send results via Telegram after video processing
- Load secrets from Infisical — never hardcode API keys or tokens
- Test regex changes against partial OCR reads (e.g., `"09:51:5"`, `"2026-01-25"`)
- Clean up orphaned snapshots when filtering episodes in `summarize()`
- When modifying `.ipynb` files, use a Python script to update the JSON programmatically
  (cannot edit `.ipynb` directly with editor tools)
- Guard ALL Colab-only imports (`google.colab`, `infisical_sdk`, `drive.mount`) with `if not RUNNING_IN_CI:` — do a full cell audit when adding CI support to any notebook
- Use `tqdm.auto` not `tqdm.notebook` — the latter crashes in CI (no widget server)
- Stitch clips only if the gap between end of one clip and start of next is ≤ 10 seconds — clips with a larger gap are separate feeding events and must each have their own FeedingTracker analysis, verdict, and Telegram block

### Never do
- Never hardcode any environment-specific identifiers (Chat IDs, Folder IDs) or secrets (Tokens, Passwords) in source code. All external identifiers MUST be loaded via `os.getenv()` or Infisical.
- Never use `show_label=True` in the video output writer (annotated video)
- Never assume class index 0 = first class in data.yaml (YOLO reorders alphabetically)
- Never use `imgsz` as a tuple — always pass a single int (e.g., `imgsz=1280`)
- Never use MixUp augmentation (destroys small kibble detail)
- Never set vertical flip augmentation (camera is fixed overhead)
- Never create new documentation files unless explicitly asked
- Never push to main/master without explicit permission
- Do not add type annotations, docstrings, or comments to code you didn't change

### Communication preferences
- Be concise — explain what changed and why, not how Python works
- Use the actual class names (Dan, Sanbo, Dan_hand, Bowl, Kibble) not generic terms
- When reporting issues, show the actual output vs expected output
- For model quality discussions, reference mAP50 and per-class AP50 numbers

---

## 13. SMOKETEST PIPELINE ARCHITECTURE

The `morning_report.ipynb` notebook uses a **5-stage pipeline**:

```
┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│ Phase 1  │  │ Phase 2  │  │Phase 2.5  │  │Phase 2.6  │  │ Phase 3   │
│ YOLO +   │─▶│Analytics │─▶│Auto-flag  │─▶│ Roboflow  │─▶│Output +   │
│ Cache    │  │(re-run!) │  │suspicious │  │ upload    │  │ Telegram  │
│ (slow)   │  │ (<2s)    │  │ (<1s)     │  │ (<10s)    │  │(save+send)│
└──────────┘  └──────────┘  └───────────┘  └───────────┘  └───────────┘
```

| Cell | ID | What it does | Speed |
|------|----|-------------|-------|
| Phase 1 | `detect-and-cache` | YOLO inference + JPEG frame cache + annotated video | Slow (minutes) |
| Phase 2 | `analyze-from-cache` | FeedingTracker with tunable params, no video I/O | Fast (<2s) |
| Phase 2.5 | `auto-flag` | Scan cache for suspicious detections (low-conf, blips, conflicts) | Fast (<1s) |
| Phase 2.6 | `roboflow-upload` | Upload flagged frames to Roboflow with tags | Fast (<10s) |
| Phase 3 | `output-and-telegram` | Save summaries, snapshots, timeline; send Telegram + flag summary | Fast (<5s) |
| Retry | `discord-notification` | Re-send to Telegram if send failed (Colab only, skipped in CI) | Fast |

### Cache format (pickle)
- `frames[i].detections` — YOLO bounding boxes
- `frames[i].timestamp` — OCR timestamp string
- `frames[i].jpeg` — compressed JPEG bytes (~50KB/frame)

### Iteration workflow for tuning
1. Run Phase 1 once (creates `_detections.pkl` cache)
2. Change thresholds in the Config cell (e.g. `SANBO_MIN_CONSECUTIVE_FRAMES`)
3. Re-run Phase 2 only — results appear in ~2 seconds
4. Repeat until satisfied, then run Phase 3 to save and send

---

## 14. LESSONS LEARNED

See [`tasks/lessons.md`](tasks/lessons.md) — updated after every correction.
