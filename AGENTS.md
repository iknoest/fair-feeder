# AGENTS.md — Fair Feeder

## 1. PROJECT OVERVIEW

**Fair Feeder** is a computer-vision cat feeding monitor that uses a Tapo IR camera
and YOLOv11 to track two cats (Dan and Sanbo), detect hand-feeding events, count
kibble, and send structured feeding reports via Telegram.

- **Target user:** Cat parent who hand-feeds Dan and wants to track how much each
  cat eats, when they arrive, and whether hand-feeding occurred.
- **Core problem:** Without monitoring, no way to know if Dan ate his fair share
  or if Sanbo stole food, especially overnight under IR lighting.

---

## 2. PRODUCT REQUIREMENTS

### Goals
- Correctly attribute kibble eaten per cat (Dan vs Sanbo) from video
- Detect Dan_hand feeding episodes with timestamps
- Send automated Telegram alerts after each video is processed
- Achieve mAP50 ≥ 0.85 on the V13 test split

### In scope
- YOLOv11 object detection (5 classes: Dan, Sanbo, Dan_hand, Bowl, Kibble)
- OCR timestamp extraction from Tapo burned-in OSD
- Phase-based eating attribution (proportional to bowl-overlap time)
- Snapshot capture: Sanbo arrival, Dan_hand episodes, kibble-dispensed moments
- Annotated video output (boxes only, no labels)
- Telegram bot notification (summary + snapshots + video)

### Out of scope
- Real-time live alerting (batch-processes recorded videos)
- Multi-camera support (single Tapo C210 only)
- Web dashboard / UI / feeding scheduling

### Non-functional
- Runs on Google Colab (free T4 GPU) — no dedicated server needed
- Inference speed < 50ms per frame at 1280px on T4
- Videos compressed with ffmpeg before Telegram upload; > 50 MB falls back to Drive
- Secrets via Infisical — never hardcoded in committed code

---

## 3. TECHNICAL ARCHITECTURE

### Tech stack
| Layer | Technology |
|-------|-----------|
| Detection model | YOLOv11s (Ultralytics), 1280px input |
| Training | Google Colab / Kaggle T4 |
| Dataset | Roboflow (ir-kibble) |
| OCR | EasyOCR (Tapo burned-in OSD) |
| Camera | Tapo C210 (RTSP + ONVIF, IR night vision) |
| Motion recording | MOG2 background subtraction (Pi 5) |
| Live cat filter | YOLOv8n @ 0.10 conf (`motion_recorder.py`) |
| Secrets | Infisical REST API |
| Notifications | Telegram Bot API |
| Storage | Google Drive (rclone on Pi, mount in Colab) |
| Automation | GitHub Actions cron → `morning_report.ipynb` via papermill |
| Experiment tracking | Weights & Biases |

### Project structure

**Root — files that must stay here (hard dependencies):**
```
fair-feeder/
├── AGENTS.md                  # This file
├── README.md                  # Project overview
├── requirements.txt           # Core dependencies
├── data.yaml                  # YOLO dataset config (5 classes) — referenced by train scripts
├── config.py                  # Camera/detection settings — imported by motion_recorder.py
├── motion_recorder.py         # 24/7 Pi daemon — path hardcoded in systemd service + SCP
├── morning_report.ipynb       # CI pipeline notebook — path hardcoded in GitHub Actions workflow
├── flagging.py                # Shared module — imported by morning_report.ipynb
├── roboflow_upload.py         # Shared module — imported by morning_report.ipynb
└── schedule_log.py            # Shared module — imported by morning_report.ipynb
```

**Subfolders:**
```
notebooks/                     # Interactive / training notebooks (run in Colab/Kaggle)
├── fair_feeder_v14.ipynb      # Current training notebook
├── smoketest.ipynb            # Inference + feeding analysis (threshold tuning)
└── batch_review.ipynb         # Historical video reprocessing

scripts/                       # One-off CLI tools (dataset prep, training, debugging)
├── train.py                   # YOLOv11 training CLI
├── download_dataset.py        # Roboflow dataset downloader
├── polygon_to_bbox.py         # Convert polygon annotations → YOLO bbox
├── verify_labels.py           # Visual label verification grid
└── debug_yolo_detection.py    # Debug YOLO detection output

deploy/                        # Pi deployment files
├── cat-monitor.service        # systemd service definition
└── sync_cleanup.sh            # Cron script to purge old local videos

tests/                         # Unit + regression tests
├── test_flagging.py           # Tests for flagging.py
├── test_roboflow_upload.py    # Tests for roboflow_upload.py
└── legacy_notebook/           # Legacy notebook regression tests

docs/                          # All documentation
├── MOTION_RECORDER_GUIDE.ipynb
├── MODELS.md                  # Model version history
├── model-improvement-handbook.md # Monthly model maintenance decision guide
├── README_RPI_SERVICE.md      # systemd setup guide
├── README_GIT_PULL.md         # Git update guide for Pi
├── blog/                      # Blog posts (EN + ZH-TW)
├── guides/                    # Pi SSH, git push guides
└── plans/                     # Design specs and implementation plans

tasks/                         # Project tracking (not code)
├── todo.md
└── lessons.md
```

**Rule:** If a file is imported by `morning_report.ipynb` or `motion_recorder.py`, or its path is hardcoded in a config/workflow, it stays at root. Everything else goes in the appropriate subfolder.

### Key dependencies
- `ultralytics` — YOLOv11 training & inference
- `roboflow` — dataset download + flagged frame upload
- `easyocr` — timestamp OCR
- `opencv-python` — video/image processing, RTSP
- `onvif-zeep-async` — ONVIF camera event subscription
- `infisicalsdk` — secret management (pip package renamed from `infisical-sdk`; import remains `from infisical_sdk`)

### External services
| Service | Auth method |
|---------|-------------|
| Roboflow | API key via Infisical |
| Infisical | `INFISICAL_ID` / `INFISICAL_SECRET` / `INFISICAL_PROJECT_ID` |
| Telegram Bot | `TelegramBotToken` + `TelegramChatId` via Infisical; `TELEGRAM_CHAT_ID` GH secret for CI |
| Google Drive | Colab `drive.mount()` (user); Service Account in CI |
| Tapo C210 | `TAPO_IP`/`TAPO_USER`/`TAPO_PASS` via Infisical or env vars |

---

## 4. CURRENT PROJECT STATUS

**Stage: Production pipeline running. V14 deployed. V15 candidate trained; validate against V14 before deployment. Data flywheel active.**

### Active surfaces
- **Pi 5** → `motion_recorder.py` (MOG2 + YOLOv8n cat filter, COCO bowl-position alert, rclone upload, 24/7 systemd)
- **GitHub Actions** → `morning_report.ipynb` via papermill, runs ~06:45 Amsterdam daily
  - Cron is `0 2 * * *` UTC to compensate observed GitHub schedule delay; workflow waits until 06:35 Europe/Amsterdam if it starts early and reports scheduler heartbeat in GitHub summary + `feeding_log.csv` (not Telegram).
- **Colab** → `smoketest.ipynb` for interactive threshold tuning; `batch_review.ipynb` for historical reprocessing
- **V14 model** → deployed baseline; historical mAP50 0.957, fresh smoketest rerun mAP50 0.690
- **V15 candidate** → trained from 155 manually revised April flagged images; fresh standalone/smoketest-style mAP50 0.741, but validate on a fixed holdout before deployment

### In progress
- [ ] **Phase C: Data Flywheel** — `docs/superpowers/specs/2026-03-26-data-flywheel-design.md`
  - [x] C1: Auto-flag + Roboflow upload (verified in CI 2026-03-26)
  - [x] C2: Batch reprocessing (231 frames uploaded with pre-annotations)
  - [x] C3: V14 trained 2026-03-28 (775 images)
  - [x] Deploy V14 to CI, update `GDRIVE_MODEL_FILE_ID` secret
  - [ ] V15 deployment decision after fixed V14/V15 validation comparison

### Planned later
- Bowl ROI zone filter in `motion_recorder.py`
- Lightweight Dan/Sanbo classifier on Pi — tag clip filenames
- Telegram-interactive flagging (reply-to-flag)

---

## 5. RASPBERRY PI 5 DEPLOYMENT (KEY FACTS)

**Production stack:** `motion_recorder.py` on systemd → MOG2 → YOLOv8n cat filter → rclone to Drive → `sync_cleanup.sh` cron deletes > 3 days old.

**Telegram commands:** `/status`, `/lastclip`, `/weight`, `/help`

**Weight tracking:** `/weight` shows inline menu — `[Log Weight] [History] [Edit]`. Log saves to `weight_log.csv` in `DRIVE_OUTPUT_DIR`, synced to Drive via rclone. History shows last 5 entries per cat + matplotlib chart (integer x-axis with MM-DD labels; falls back to text-only on ImportError). 30-day reminder sent via Telegram by morning_report.ipynb if no weight logged in 30 days. `/syncstatus` merged into `/status` — Drive file count appended via `rclone size` (8s timeout).

**Deployment expectation:** When changing `motion_recorder.py` or other Pi-runtime files, Codex should deploy to the Pi and restart `cat-monitor.service` autonomously unless the user explicitly says not to. Standard flow: SCP the changed file to `/home/pi5/Feeder/fair-feeder/`, run Pi-side `py_compile`, restart the service, then verify `systemctl is-active cat-monitor.service` and recent logs/status.

### Live constraints
- RTSP **must** use TCP transport (UDP unreliable on Pi 5)
- YOLOv8n at conf 0.10 — older EfficientDet hallucinated bboxes at ground-level camera angles
- `infisical-sdk` has no ARM64 wheel → use Infisical Universal Auth REST API via `requests`
- `ai-edge-litert` API unstable on Pi — YOLOv8n more reliable
- Drive uses `rclone copy` (not `bisync`) to avoid `.lck` deadlocks
- 2304×1296 @ 15fps ≈ 50–100 MB/min — save only cat-positive clips (~10% of motion events)

### Pi 5 vs Colab roles
| Task | Pi 5 | Colab / CI |
|------|------|------------|
| 24/7 motion recording | ✅ | ❌ |
| YOLOv11 analysis | ❌ (CPU bound) | ✅ |
| Report generation | ❌ | ✅ |

---

## 6. SMOKETEST PIPELINE ARCHITECTURE

`morning_report.ipynb` is a **5-stage pipeline**:

| Phase | Cell ID | What it does | Speed |
|-------|---------|-------------|-------|
| 1 | `detect-and-cache` | YOLO inference + JPEG frame cache + annotated video | Slow (minutes) |
| 2 | `analyze-from-cache` | FeedingTracker with tunable params, no video I/O | Fast (<2s) |
| 2.5 | `auto-flag` | Scan cache for low-conf / blips / conflicts | Fast (<1s) |
| 2.6 | `roboflow-upload` | Upload flagged frames with tags | Fast (<10s) |
| 3 | `output-and-telegram` | Save summaries, snapshots, timeline; send Telegram | Fast (<5s) |

### Cache format (pickle)
- `frames[i].detections` — YOLO bboxes
- `frames[i].timestamp` — OCR string
- `frames[i].jpeg` — compressed JPEG bytes (~50KB/frame)

### Tuning workflow
1. Run Phase 1 once (creates `_detections.pkl`)
2. Change thresholds in Config cell
3. Re-run Phase 2 only (~2s)
4. Phase 3 to save and send

---

## 7. CODING CONVENTIONS

### Style
- Python, **no type annotations** unless already present in the edited file
- Use existing helpers (`draw_boxes`, `bbox_iou`, `parse_results`) — don't create duplicates
- Notebook cells self-contained where possible (re-scan `SOURCE_DIR`, etc.)
- Class colours → `CLASS_COLORS_RGB` + `get_color_bgr()`
- Detection thresholds live in the Config cell — never scatter magic numbers

### Always do
- Read a file before editing — understand existing code first
- Preserve 16:9 aspect ratio (`rect=True`) in all YOLO calls
- Use `model.names` dict for class index mapping (YOLO sorts alphabetically)
- Keep annotated video boxes-only (`show_label=False`) — owner explicitly requested this
- Load secrets via Infisical / `os.getenv()` — never hardcode
- Test regex changes against partial OCR reads (e.g. `"09:51:5"`, `"2026-01-25"`)
- Guard Colab-only imports (`google.colab`, `infisical_sdk`, `drive.mount`) with `if not RUNNING_IN_CI:` — full cell audit when adding CI support
- Use `tqdm.auto` not `tqdm.notebook` (the latter crashes in CI — no widget server)
- When editing `.ipynb`, update the JSON programmatically via a Python script (cannot edit directly with editor tools); always strip `\r` on Windows
- Stitch clips only if gap ≤ 10 seconds — larger gaps are separate feeding events, each with its own FeedingTracker analysis and Telegram block

### Never do
- Hardcode environment-specific identifiers (Chat IDs, Folder IDs) or secrets
- `show_label=True` in annotated video writer
- Assume class index 0 = first class in `data.yaml`
- Pass `imgsz` as a tuple — always a single int
- MixUp augmentation (destroys small kibble detail)
- Vertical flip augmentation (camera is fixed overhead)
- Create new documentation files unless explicitly asked
- Add type annotations, docstrings, or comments to code you didn't change

### Commit / push automation
- For CI-facing or Pi-runtime fixes, Codex should do the security check, file review, code cleanup, verification, commit, and push to `main` automatically after the work is ready.
- Before pushing: review `git status`/diff, exclude unrelated user files, scan for secrets or hardcoded environment IDs, run focused tests and compile/notebook checks, and confirm the pushed commit landed on `origin/main`.
- If a push would include destructive history changes, credentials, or unrelated user work, stop and ask first.

### Communication preferences
- Be concise — what changed and why, not how Python works
- Use actual class names (Dan, Sanbo, Dan_hand, Bowl, Kibble), not generic terms
- Show actual vs expected output when reporting issues
- Reference mAP50 / per-class AP50 for model quality discussions

---

## 8. DOCUMENTATION UPDATES

When the user asks to document a lesson, decision, or fix, update **all four** in the same change:
- `AGENTS.md` — if it affects how Codex should work on the project
- `tasks/lessons.md` — the generalised anti-pattern rule
- `tasks/todo.md` — task state
- `README.md` — if user-facing behaviour changed

**Never** update one without checking the others. Past omission: updating lessons.md + todo.md but forgetting AGENTS.md.

**File roles:** AGENTS.md = what is true about this project · lessons.md = what went wrong and how to avoid it · todo.md = what needs to be done.

---

## 9. CI/CD DEBUGGING

Before pushing a GitHub Actions fix, do a **full pre-flight audit** — don't rely on fix-push-wait-fail cycles. The morning-report workflow has burned us on missing env vars, Drive 403, `tqdm.notebook` widgets, and CSV create-vs-update. Each of these was fixable locally.

### Pre-flight checklist
- [ ] Every secret the notebook reads is listed under `env:` in the workflow step (not just in repo settings)
- [ ] All imports work headless — no `tqdm.notebook`, no `ipywidgets`, no `google.colab` outside a `RUNNING_IN_CI` guard
- [ ] Service account has scope for every Drive file it touches; uses `update()` not `create()` (SA has zero quota on personal Drive)
- [ ] Datetimes filtered with `pytz.timezone('Europe/Amsterdam')` — never naive UTC
- [ ] System deps (ffmpeg) installed via `apt-get` step before pip
- [ ] Execute the notebook locally with `jupyter nbconvert --execute` or `papermill` first

When a CI run fails, list **every** likely failure mode before pushing — fix them all in one commit, not one at a time.

---

## 10. GOOGLE DRIVE ARCHIVAL

Archive pipeline outputs **per-run with timestamped filenames**, never overwrite a single file. Previous mistake: proposing overwrite-the-canonical-file, which loses history on every run.

**Exception:** `feeding_log.csv` — always download current file from Drive (`get_media()`), remove today's row if present (dedup), append fresh row, then `update()`. Do **not** call `create()` from CI (service account has zero quota). Columns: `date, dan_kibble, sanbo_kibble, hand_feeding, compensation, video_count, dan_first_arrival, sanbo_first_arrival, schedule_time, start_time, flagged_frames, roboflow_uploaded, roboflow_skipped, roboflow_failed, flag_top_tags, dan_weight, sanbo_weight`. `schedule_time` and `start_time` are Europe/Amsterdam local time with DST handled by timezone conversion. Weight columns are backfilled from `weight_log.csv` (Pi-generated, lives in `GDRIVE_UPLOAD_FOLDER_ID`); schedule/start columns are backfilled from GitHub Actions run history when `GITHUB_TOKEN` is available.

**CI Drive upload policy:** Large binary outputs (videos, archives) are **not** uploaded from CI — the service account has no storage quota. Colab (user account) handles archive. Telegram already delivers daily results.

---

## 11. CORE PRINCIPLES

- **Simplicity first** — make every change as simple as possible. Impact minimal code.
- **Root cause only** — no temporary fixes. Find the actual issue.
- **Minimal blast radius** — changes touch only what's necessary.
- **Plan before non-trivial work** — enter plan mode for 3+ step tasks or architectural decisions. If something goes sideways, **stop and re-plan**.
- **Verify before "done"** — run tests, check logs, diff behaviour. Never mark complete without proof.
- **Autonomous bug fixing** — given logs + symptoms, just fix it. Don't ask for hand-holding.

---

## 12. RECENT ISSUES (pattern-learning value)

Full history in git log. These are the ones whose pattern keeps catching us:

| # | Issue | Root cause | Fix |
|---|-------|-----------|-----|
| 34 | `feeding_log.csv` not accumulating / wrong kibble count / duplicates on manual trigger | CSV only read last event; no dedup for same-day runs | Aggregate all `video_results`; dedup by removing today's row before appending; new columns: arrivals + weight |
| 35 | GitHub Actions scheduled workflows may start hours after cron | Runs scheduled at 04:45 UTC were actually starting around 09:08-09:19 Amsterdam; later `0 3 * * *` UTC runs still started after 08:00 Amsterdam on multiple days. Cron time alone is not reliable. | Schedule early (`0 2 * * *` UTC), wait until feeding window close if the runner starts early, and record scheduler heartbeat in GitHub summaries + `feeding_log.csv`. Keep Telegram concise. |
| 33 | Annotated video never appears in Drive from CI | SA has zero storage quota; `files().create()` 403 | Dropped Drive uploads from CI; Colab archives |
| 32 | Telegram sent unmerged short clip instead of merged | Phase 1/2 re-scanned `SOURCE_DIR`, overwriting stitch output | Guarded rescan behind `if not RUNNING_IN_CI:` |
| 31 | FeedingTracker reports "0 kibble / no activity" despite timeline showing kibble | `_find_clear_kibble_count` searches no-cat frames; model only detects kibble when cats present | Added phase-entry/exit fallback methods |
| 30 | "Today" filter misses morning captures | Naive UTC datetimes — 07:00 CET → previous day UTC | Always use `pytz.timezone('Europe/Amsterdam')` |
| 29 | `IndentationError` in papermill after editing `.ipynb` on Windows | `\r\n` carriage returns break IPython lexer when passed to Jupyter JSON | Always `.replace('\r', '')` when pushing into `cell['source']` |
| 21 | CSV creation raised `403 storageQuotaExceeded` | Service accounts have zero quota on personal Drive | Pre-create in UI, share with SA, CI only calls `update()` |
| 20 | `tqdm.notebook` ImportError in CI | `tqdm.notebook` requires `ipywidgets`/IntProgress; no widget server in papermill | Replaced with `tqdm.auto` everywhere |
| 19 | Silent empty-secret CI runs | Secrets set in repo settings but not listed under step `env:` | List every secret explicitly under `env:` |
| 17 | Recording stops/restarts during continuous motion | Tapo ONVIF sends events in bursts with 1–3s gaps | Use `last_motion_time` timer, stop after 5s with no event |
| 6 | Eating attribution double-counted across overlapping phases | Per-episode accounting | Rewrote with phase-based attribution + double-counting guard using `peak_kibble` |
| 4 | `Dan_hand` false positives without Dan body present | No co-detection requirement | `dan_here` check: `Dan_hand` requires Dan body in same frame |

---

## 13. ACTIVE DESIGN DECISIONS (non-obvious)

| Decision | Reasoning |
|----------|-----------|
| YOLOv11s (not m/l) | 775 training images is too few for 20M+ params. Small object detection comes from 1280px input, not model size. |
| 1280px inference | Kibble is tiny; 640px default misses too many |
| `rect=True` everywhere | Preserves Tapo's 16:9 aspect ratio; letterbox distorts cat proportions |
| Phase-based eating attribution | Correctly handles overlapping feeding (both cats at bowl); per-episode counting double-counts shared phases |
| Rolling median (window=3) for kibble smoothing | Simple, removes single-frame flicker without losing real transitions |
| `Dan_hand` requires Dan body co-detection | Hand can't exist without the cat; eliminates stray false positives |
| `peak_kibble = max(counts)` for double-counting guard | `first_clear` underestimates starting kibble (~30-40% occlusion when cat already present) |
| Boxes-only annotated video | Owner explicitly requested no labels — labels + percentages obscure kibble |
| Stitch clips only if gap ≤ 10s | Larger gaps are genuinely separate feeding events; merging breaks FeedingTracker attribution |
| Each distinct event gets its own report | Per-event kibble / verdict / Telegram block, not combined |
| `_fmt_time()` strips date when same as video start | Redundant dates clutter mobile Telegram bubble; date is in the header |
| Action verdict is the first line of the Telegram message | Telegram push notification shows the first line — owner needs to know instantly whether action is required without opening the message. Format: `😸 Dan finished breakfast`, `😿 Give Dan ~N kibble`, or `🍽️? Feeding machine not working?`. Do not include schedule delay or explanatory `Why:` lines in Telegram; keep scheduler heartbeat in logs/GitHub summaries. |
| Episode numbers are continuous across clips (day-wide offset) | Each `FeedingTracker` receives an `episode_offset` = sum of confirmed episodes from all prior clips. Snapshot keys `dan_hand_epN` / `kibble_dispensed_epN` use day-wide N, not per-clip N. |
| `kibble_dispensed` snapshot prefers stable pre-cat kibble | Use a stable no-cat frame before cats cover the bowl when available, so Telegram shows inspectable kibble. If no clean pre-cat frame exists, wait for 3 consecutive clear frames after Dan_hand; final fallback is the highest-kibble frame within 5s. |
| Bowl-position alert checks full visibility, not frame center | The bowl normally sits on the right side of the Tapo frame. Alert only when the bowl is missing or its bbox is clipped/not fully visible; a best-center like 78%,61% can be a good position. Missing bowl alerts start with `🥣?`; clipped/not-visible camera-position alerts start with `👀?`. |
| Compensation = `sanbo_kibble_eaten` | Directly answers "how much extra does Dan need?" |
| ffmpeg compression (crf=28, 720p) for Telegram | Most feeding videos fit under 50 MB inline; preserves inline playback |
| `RUNNING_IN_CI` guard for Colab-only cells | Single env-check flag is cleaner than try/except per-import or duplicate notebooks |
| `tqdm.auto` over `tqdm.notebook` | `notebook` crashes in CI; `auto` works everywhere |
| SA `update()` not `create()` for Drive | SA has zero storage quota; `create()` fails with 403 |
| CI = cron only; Colab = interactive | CI handles daily scheduled runs; threshold tuning and ad-hoc checks in Colab |
| Roboflow pre-annotations (`is_prediction=True`) | Reviewer corrects model output — faster than labeling from scratch |
| Monthly Roboflow batches (`flagged-YYYY-MM`) | ~12 batches/year is manageable vs 180 daily |
| `roboflow_uploaded.txt` on Drive for dedup | Simple append-only file prevents duplicate uploads across batch sessions |
| Tapo OCR: replace `\|:` → `:1` **before** `\|` → `1` | EasyOCR reads `:1` as `\|:` from Tapo's thin OSD; order matters (single replace gives extra colon) |
| Copy-paste augmentation off for V14 | Kibble already dominates (4015 annotations); data bottleneck is Sanbo (293) and Dan (438) |
| `last_motion_time` timer for recording stop | Tapo ONVIF sends event bursts with gaps; instantaneous flag causes premature stops |

---

## 14. LESSONS LEARNED

See [`tasks/lessons.md`](tasks/lessons.md) — updated after every correction.
