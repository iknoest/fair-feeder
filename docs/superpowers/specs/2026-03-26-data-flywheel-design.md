# Data Flywheel — Continuous Model Improvement

**Date**: 2026-03-26
**Status**: Approved, pending implementation
**Scope**: Phase C — auto-flagging, Roboflow upload, batch reprocessing, retraining cycle

---

## 1. Problem

The YOLOv11 model (v13) has been running in production for 2 days. Early results show false positives (e.g., Sanbo hallucinated when not present). With ~20 historical Pi-captured videos unanalyzed, there's no feedback loop from daily outputs back to model improvement.

The Service Account (SA) used in CI has zero Google Drive storage quota, blocking `files().create()` for new outputs. This design eliminates the Drive upload dependency from CI entirely.

## 2. Goals

- Automatically detect suspicious model predictions from each daily run
- Upload flagged frames to Roboflow for human review and relabeling
- Enable batch reprocessing of historical Pi videos in Colab
- Establish a sustainable retrain cadence (~biweekly)
- Require minimal daily effort from the user (30 seconds Telegram check)

## 3. Architecture

### 3.1 Daily CI pipeline (GitHub Actions)

```
Phase 1:   YOLO inference → detection cache (/tmp/)
Phase 2:   FeedingTracker analytics
Phase 2.5: Auto-flag suspicious detections from cache       ← NEW
Phase 2.6: Upload flagged frames → Roboflow (API)           ← NEW
Phase 3:   Send report + flag summary → Telegram
```

Drive uploads are **removed from CI**. Telegram remains the daily delivery channel. Roboflow receives flagged frames directly via API (no Drive involved).

### 3.2 Colab batch reprocessing

New notebook `batch_review.ipynb`:
1. Mount Drive
2. Scan video folder for all clips (no feeding window filter)
3. Run Phase 1 → Phase 2 → Phase 2.5 → Phase 2.6 per video
4. Save annotated videos + timelines to Drive directly (user account, no SA quota issue)
5. Print summary: "Processed N videos, flagged M frames → Roboflow"

### 3.3 Roboflow review + retrain

1. User reviews flagged frames in Roboflow UI (filter by tag)
2. Correct labels, assign to training split
3. When ~50-100 corrections accumulate: export new dataset version, retrain in Colab
4. Deploy new model weights via `GDRIVE_MODEL_FILE_ID` secret

## 4. Auto-flagging logic (Phase 2.5)

Scans the detection cache (already contains per-frame YOLO results with confidence scores and JPEG frames). Flags frames matching any of these criteria:

### 4.1 Flag criteria

| # | Criterion | Logic | Example |
|---|-----------|-------|---------|
| 1 | **Low-confidence detection** | Any detection with confidence < `FLAG_CONF_THRESHOLD` (default: 0.40) | Sanbo at 0.31 confidence |
| 2 | **Single-frame blip** | A class appears for <=2 consecutive frames then disappears for >=5 frames | Dan_hand flashes for 1 frame |
| 3 | **No co-detection** | Dan_hand detected without Dan body in same frame | Hand without cat |
| 4 | **High-confidence conflict** | Two cat classes (Dan, Sanbo) overlap >50% IoU in same frame | Both boxes on same cat |
| 5 | **Kibble count jump** | Kibble count changes by >5 between consecutive frames | 3 → 11 in one frame |

### 4.2 Deduplication

If multiple criteria flag the same frame, it's uploaded once with all applicable tags. Adjacent flagged frames (within 3 frames of each other) are deduplicated — only the frame with the highest-confidence detection is kept, to avoid uploading 15 near-identical frames.

### 4.3 Configuration

All thresholds live in the notebook config cell:

```python
FLAG_CONF_THRESHOLD = 0.40      # Below this = low confidence flag
FLAG_BLIP_MAX_FRAMES = 2        # Appears for <= this many frames = blip
FLAG_BLIP_GAP_FRAMES = 5        # Must disappear for >= this many frames after
FLAG_IOU_CONFLICT = 0.50        # Dan/Sanbo overlap threshold
FLAG_KIBBLE_JUMP = 5            # Kibble count change threshold
FLAG_DEDUP_WINDOW = 3           # Adjacent flagged frames within this window → keep best
```

## 5. Roboflow upload (Phase 2.6)

### 5.1 API setup

Roboflow Upload API requires:
- **API key**: Already stored in Infisical as `ROBOFLOW_API_KEY`
- **Workspace name**: The user's Roboflow workspace slug
- **Project name**: `ir-kibble` (existing project)

New GitHub Actions secret required:
- `ROBOFLOW_API_KEY` — Roboflow API key (from Roboflow Settings → API Keys)

### 5.2 Upload format

```python
from roboflow import Roboflow
import tempfile, os

ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]
ROBOFLOW_WORKSPACE = "<workspace-slug>"   # from Roboflow URL
ROBOFLOW_PROJECT = "ir-kibble"

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)

def upload_flagged_frame(jpeg_bytes, filename, tags, batch_name):
    """Upload a single flagged frame to Roboflow."""
    # SDK requires a file path, so write JPEG bytes to a temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(jpeg_bytes)
        tmp_path = f.name
    try:
        project.upload(
            image_path=tmp_path,
            batch_name=batch_name,
            tag_names=tags,          # list of strings
            # omit split → lands in "Unassigned" pool in Roboflow UI
            num_retry_uploads=2,
        )
    finally:
        os.unlink(tmp_path)
```

> **Note**: Omitting the `split` parameter places images in Roboflow's "Unassigned" pool.
> The user assigns them to train/valid/test after review. Multiple tags are passed as a
> list via `tag_names`.

### 5.3 Naming convention

- **Filename**: `{video_stem}_frame{N:05d}.jpg` (e.g., `merged_20260326_0618_frame00142.jpg`)
- **Batch**: `flagged-YYYY-MM` (e.g., `flagged-2026-03`)
- **Tags**: One or more from the tag table below

### 5.4 Tag definitions

| Tag | Meaning |
|-----|---------|
| `low-conf-dan-{score}` | Dan detected below FLAG_CONF_THRESHOLD. Score is 2-digit int (e.g., `low-conf-dan-31` = 0.31) |
| `low-conf-sanbo-{score}` | Sanbo detected below threshold |
| `low-conf-dan_hand-{score}` | Dan_hand detected below threshold |
| `low-conf-kibble-{score}` | Kibble detected below threshold |
| `low-conf-bowl-{score}` | Bowl detected below threshold |
| `blip-dan` | Dan appeared for <=2 frames then vanished |
| `blip-sanbo` | Sanbo appeared for <=2 frames then vanished |
| `blip-dan_hand` | Dan_hand flash |
| `blip-kibble` | Kibble flash |
| `no-codetect-dan_hand` | Dan_hand without Dan body in frame |
| `conflict-dan-sanbo` | Dan and Sanbo boxes overlap >50% IoU |
| `kibble-jump-{delta}` | Kibble count changed by delta in one frame (e.g., `kibble-jump-8`) |

### 5.5 Error handling

- If Roboflow upload fails (network error, quota), log warning and continue. Never block the daily pipeline.
- Telegram report still sends regardless of Roboflow upload success.
- Upload failures are reported in Telegram: `"⚠️ Roboflow upload failed for 3/12 frames"`.

## 6. Telegram report enhancement

Add a new line at the end of the daily Telegram summary:

```
🔍 Auto-flagged: 12 frames → Roboflow
   3× low-conf-sanbo, 7× blip-kibble, 2× kibble-jump
```

If zero frames flagged:
```
🔍 No suspicious detections flagged
```

If Roboflow upload partially failed:
```
🔍 Auto-flagged: 12 frames → Roboflow (9 uploaded, ⚠️ 3 failed)
   3× low-conf-sanbo, 7× blip-kibble, 2× kibble-jump
```

## 7. Batch review notebook (`batch_review.ipynb`)

Separate Colab notebook for reprocessing historical videos.

### 7.1 Cells

| Cell | Purpose |
|------|---------|
| 0 | Mount Drive, install dependencies, load model |
| 1 | Config: video folder path, output folder path, flagging thresholds |
| 2 | Scan video folder, list all clips, show count |
| 3 | Phase 1: YOLO inference + cache (loop over all videos) |
| 4 | Phase 2: FeedingTracker per video |
| 5 | Phase 2.5: Auto-flag suspicious detections |
| 6 | Phase 2.6: Upload flagged frames to Roboflow |
| 7 | Save outputs: annotated videos + timelines + summaries to Drive |
| 8 | Summary: total videos processed, total frames flagged, per-video breakdown |

### 7.2 Shared code

The auto-flagging logic (Phase 2.5) and Roboflow upload (Phase 2.6) are identical between `morning_report.ipynb` and `batch_review.ipynb`. Extract into a shared Python module:

```
fair-feeder/
├── flagging.py          # auto_flag_detections(cache) → list of flagged frames
├── roboflow_upload.py   # upload_flagged_frames(frames, api_key, project, batch)
```

Both notebooks import from these modules. Single source of truth for flagging logic and upload code.

## 8. CI changes

### 8.1 Remove Drive video upload from `morning_report.ipynb`

Delete the annotated video `files().create()` call in Cell 14. Keep:
- CSV append + `files().update()` (already working)
- SA auth for reading input videos (already working)

### 8.2 New GitHub Actions secret

Add `ROBOFLOW_API_KEY` to repository secrets. Add to workflow env:

```yaml
env:
  ROBOFLOW_API_KEY: ${{ secrets.ROBOFLOW_API_KEY }}
```

### 8.3 Roboflow workspace/project config

Add to notebook config cell:
```python
ROBOFLOW_WORKSPACE = "<workspace-slug>"
ROBOFLOW_PROJECT = "ir-kibble"
```

## 9. Retraining cycle

| Activity | Frequency | Time |
|----------|-----------|------|
| Check Telegram report | Daily | 30 seconds |
| Review flagged frames in Roboflow | Weekly | 15-30 min |
| Retrain model (Colab) | Every 2-4 weeks (when ~50-100 corrections accumulate) | ~1 hour (GPU time) |

### 9.1 Retrain steps

1. Open Roboflow → Annotate tab → filter by `flagged-YYYY-MM` batch
2. Review and correct labels for flagged images
3. Assign corrected images to training split
4. Generate new dataset version (v14, v15, ...)
5. Open `fair_feeder_v13.ipynb` in Colab → change version number → train
6. Compare mAP50 against previous version
7. If improved: upload new weights to Drive, update `GDRIVE_MODEL_FILE_ID` secret

## 10. Roboflow API setup guide

### Step-by-step for the user

1. **Find your API key**:
   - Go to [roboflow.com](https://roboflow.com) → Settings (gear icon) → Roboflow API → Copy "Private API Key"

2. **Find your workspace slug**:
   - Look at your Roboflow URL: `https://app.roboflow.com/<workspace-slug>/ir-kibble`
   - The part after `app.roboflow.com/` and before `/ir-kibble` is your workspace slug

3. **Add as GitHub secret**:
   - Go to your `fair-feeder` repo → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `ROBOFLOW_API_KEY`
   - Value: paste your API key
   - Click "Add secret"

4. **Store in Infisical** (for Colab use):
   - Add `ROBOFLOW_API_KEY` to your Infisical project (same place as Telegram tokens)
   - The batch_review notebook will load it from Infisical via the existing auth pattern

## 11. Out of scope

- Real-time Telegram flagging (Approach 3 from brainstorming — can be layered later)
- Automatic retraining (always manual — user decides when quality is sufficient)
- Active learning model selection (Roboflow has this feature but adds complexity)
- Multi-model comparison pipeline (one model at a time is enough)

## 12. Success criteria

- [ ] Daily CI run uploads flagged frames to Roboflow without errors
- [ ] Telegram report shows flagged frame count and tag breakdown
- [ ] 20 historical videos reprocessed via batch_review.ipynb
- [ ] At least one retrain cycle completed (v14) with improved mAP50
- [ ] Drive video upload removed from CI (no more 403 errors)
