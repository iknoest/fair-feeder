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
| Motion recording | ONVIF PullPoint events | Camera-side motion detection triggers recording |
| Live detection | YOLOv8n (0.10 conf) | Local real-time cat detection (`motion_recorder.py`) |
| Cat identification | Custom histogram analysis | Distinguishes Dan (tuxedo/dark) from Sanbo (calico/orange) |
| Secret management | Infisical SDK | Stores Roboflow key, Telegram credentials |
| Notifications | Telegram Bot API | Sends summaries, photos, video to owner's phone |
| Storage | Google Drive (mounted as H:\) | Persistent storage for models, videos, outputs |
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
├── test_yolo_detection.py     # Utility to debug YOLO on Pi camera
└── tasks/
    ├── todo.md                # Current task tracking (checkable items)
    └── lessons.md             # Self-improvement log (updated after corrections)
```

### Key dependencies
- `ultralytics` — YOLOv11 training & inference
- `roboflow` — dataset download
- `easyocr` — timestamp OCR
- `opencv-python` — video/image processing, RTSP frame reading, recording
- `mediapipe` — real-time detection (main.py, motion_recorder.py cat filter)
- `onvif-zeep-async` — ONVIF camera event subscription (motion detection)
- `infisical-sdk` — secret management
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
- [x] YOLOv11 V13 model trained (5 classes, mAP50=0.928 for Dan_hand)
- [x] Smoketest notebook: full video analysis pipeline (requires GPU, runs on Colab)
- [x] FeedingTracker: phase-based kibble attribution
- [x] Telegram bot integration with video uploads
- [x] All AI analysis features (timestamp OCR, kibble counting, hand-feeding detection)

### In progress
- [ ] **Fix cat detection on Pi 5** - Debug ai-edge-litert API in vision/detector.py
- [ ] **24/7 systemd service** - Enable motion_recorder.py or test_motion_pi.py as background service
- [ ] Testing model on more real-world videos

### Planned next
1. Fix ai-edge-litert API issue OR use Haar Cascade fallback for cat detection
2. Deploy as systemd service for continuous monitoring
3. Monitor storage and verify rclone uploads working

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
  - Use rclone to upload immediately, delete local copy
  - Monitor: `df -h /home/pi5/Pictures/gdrive-randomdice-sync/`

#### 3. **Python Environment Fragmentation**
- **Problem**: Different dependencies work on different platforms
  - `mediapipe` not available on Pi (ARM build unavailable)
  - `tensorflow` / `tflite-runtime` wheel availability varies
  - `ai-edge-litert` is new (2024) — fewer legacy issues but unstable API
- **Solution**: Use lightweight alternatives
  - OpenCV Haar Cascade (built-in, no external ML framework needed)
  - Test on Pi before committing to large packages

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
| **ai-edge-litert API crash** | crashes during `interpreter.invoke()` | Fix vision/detector.py to match actual API |
| **Virtual environment bloat** | pip install hangs on ARM wheels | Pre-filtered requirements.txt for Pi |
| **Path separators (W vs U)** | `H:\` Drive letter doesn't exist | Use `platform.system()` checks |
| **File encoding (UTF-16 logs)** | PowerShell output incompatible | Use Python `open()` with utf-8 explicit |
| **Google Drive not mounted** | "Permission denied" on /gdrive | Use rclone instead of Drive API |
| **Credentials in env vars** | Real password leaked in .py files | Use config.py + Infisical fallback |

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
├─ Uploads to Google Drive (rclone)
└─ Deletes local copy when uploaded
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

### Unresolved
- **Detection model quality on real videos** — only 2 Colab test videos run; need validation on Pi-recorded footage

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

### Still open (nice-to-have)
- Scheduling implementation details (cron, Cloud Functions, etc.)

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

### Never do
- Never hardcode API keys, tokens, or passwords in committed code
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

The `smoketest.ipynb` notebook uses a **3-stage pipeline** for efficient iteration:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Phase 1    │     │  Phase 2     │     │  Phase 3        │
│  YOLO +     │────▶│  Analytics   │────▶│  Output +       │
│  Cache      │     │  (re-run!)   │     │  Telegram       │
│  (slow)     │     │  (<2s)       │     │  (save & send)  │
└─────────────┘     └──────────────┘     └─────────────────┘
```

| Cell | ID | What it does | Speed |
|------|----|-------------|-------|
| Phase 1 | `detect-and-cache` | YOLO inference + JPEG frame cache + annotated video | Slow (minutes) |
| Phase 2 | `analyze-from-cache` | FeedingTracker with tunable params, no video I/O | Fast (<2s) |
| Phase 3 | `output-and-telegram` | Save summaries, snapshots, timeline; send Telegram | Fast (<5s) |
| Retry | `discord-notification` | Re-send to Telegram if send failed | Fast |

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
