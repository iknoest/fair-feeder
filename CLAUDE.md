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
- Video files up to 50 MB can be sent via Telegram; larger ones fall back to Drive path
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
| Camera | Tapo C210 (RTSP) | IR night vision, 2K resolution, affordable |
| Live detection | MediaPipe EfficientDet Lite2 | Local real-time cat detection (main.py) |
| Cat identification | Custom histogram analysis | Distinguishes Dan (tuxedo/dark) from Sanbo (calico/orange) |
| Secret management | Infisical SDK | Stores Roboflow key, Telegram credentials |
| Notifications | Telegram Bot API | Sends summaries, photos, video to owner's phone |
| Storage | Google Drive | Persistent storage for models, videos, outputs |
| Experiment tracking | Weights & Biases | Training metrics, loss curves, checkpoints |

### Project structure
```
fair-feeder/
├── CLAUDE.md                  # This file
├── config.py                  # Camera, detection, identification settings
├── main.py                    # Live monitoring (Tapo RTSP → MediaPipe → display)
├── train.py                   # YOLOv11 training CLI
├── download_dataset.py        # Roboflow dataset downloader
├── polygon_to_bbox.py         # Convert polygon annotations → YOLO bbox
├── verify_labels.py           # Visual label verification grid
├── tapo_check.py              # RTSP connection tester
├── test_env.py                # Environment validation
├── data.yaml                  # YOLO dataset config (5 classes)
├── requirements.txt           # Core dependencies
├── fair_feeder_v13.ipynb      # Training notebook (Colab/Kaggle)
├── smoketest.ipynb            # Inference + feeding analysis notebook
├── efficientdet_lite0.tflite  # MediaPipe model (7 MB, lighter)
├── efficientdet_lite2.tflite  # MediaPipe model (12 MB, primary)
└── vision/
    ├── __init__.py
    ├── detector.py            # MediaPipe object detection wrapper
    ├── identifier.py          # Cat identification (Dan vs Sanbo)
    └── motion.py              # Background subtraction (MOG2)
```

### Key dependencies
- `ultralytics` — YOLOv11 training & inference
- `roboflow` — dataset download
- `easyocr` — timestamp OCR
- `opencv-python` — video/image processing
- `mediapipe` — real-time detection (main.py)
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

**Stage: Late prototype / early MVP**

### Completed
- [x] YOLOv11 V13 model trained (5 classes, mAP50=0.928 for Dan_hand)
- [x] Smoketest notebook: full video analysis pipeline
- [x] FeedingTracker: phase-based kibble attribution
- [x] OCR timestamp extraction (with partial-second handling)
- [x] Snapshot capture: Sanbo arrival, Dan_hand episodes, kibble-dispensed
- [x] Annotated video output (boxes only, no labels)
- [x] Text summary saved to Google Drive
- [x] Telegram bot integration (summary + snapshots + video)
- [x] Infisical secret management
- [x] Kibble count smoothing (rolling median, window=3)
- [x] Dan_hand co-detection (requires Dan body in same frame)
- [x] Orphaned snapshot cleanup in summarize()
- [x] Production acceptance criteria documented in notebook
- [x] Live monitoring with MediaPipe (main.py)

### In progress
- [ ] Testing model on more real-world videos (owner ran 2 so far)
- [ ] Evaluating detection quality against 8-scenario checklist

### Planned next (prioritized)
1. Run 5–10 more test videos across different scenarios (IR, motion blur, two cats)
2. Evaluate if model needs retraining with more examples
3. Move Tapo camera credentials out of config.py into Infisical
4. Automate video-to-analysis pipeline (currently manual notebook runs)

---

## 5. KNOWN LIMITATIONS

### Technical constraints
- **EasyOCR sometimes drops digits** — the seconds field may show 1 digit
  instead of 2. Current fix: zero-pad partial seconds and mark with `?` if ambiguous.
- **Kibble count flickers** across frames as kibble pieces shift in the bowl.
  Mitigated with rolling median (window=3), but not eliminated.
- **Dan_hand detection requires Dan body in frame** — if the hand enters
  frame before the cat body, it won't be detected until both are visible.
- **Telegram video limit is 50 MB** — longer videos exceed this and only
  a Drive path is sent instead.
- **FeedingTracker stores frame copies in memory** for kibble-dispensed
  snapshots, which increases RAM usage during long videos.

### Scope limitations
- Only two cats supported (Dan and Sanbo) — class IDs are hardcoded
- Only one camera angle (fixed overhead Tapo C210)
- Batch processing only (no real-time feeding alerts from smoketest pipeline)
- No web UI — all interaction is via Colab notebooks + Telegram

### Platform / environment
- Training and inference require Google Colab with GPU (T4)
- Live monitoring (main.py) requires local machine with Python + webcam or Tapo RTSP access
- Google Drive must be mounted for file I/O in notebooks

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

### Unresolved
- **Detection model quality** — only 2 test videos run so far; need 5–10 across
  all 8 scenarios before declaring production-ready
- **Tapo camera password hardcoded** in `config.py` line 21 as fallback default —
  should be moved to Infisical or `.env`

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
- Performance optimization for videos > 50 MB (current Telegram limit)

---

## 9. CONVENTIONS & RULES FOR AI

### Coding style
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
