# Monthly Retraining Procedure

Use this checklist at the start of each month, or whenever Roboflow has enough corrected flagged frames to justify a new model.

## 1. Review Model Drift

Check the recent `feeding_log.csv` rows and Telegram flag summaries.

Key columns:
- `flagged_frames`: daily number of suspicious frames.
- `roboflow_uploaded`: new flagged frames sent to Roboflow.
- `roboflow_skipped`: frames already uploaded before.
- `roboflow_failed`: upload failures.
- `flag_top_tags`: recurring failure modes.

Direction guide:
- `conflict-dan-sanbo`: improve Dan/Sanbo identity and overlap examples.
- `low-conf-dan` or `low-conf-sanbo`: add more varied cat pose and lighting examples.
- `blip-dan`, `blip-sanbo`, `blip-dan_hand`: check one-frame hallucinations.
- `no-codetect-dan_hand`: correct hand-feeding labels and ensure Dan body is labeled.
- `kibble-jump-*`: only high-value when the bowl is clear; skip normal cat-eating occlusion.
- camera-position alerts: exclude bad camera days or label only if the bowl is visible and useful.

## 2. Review Roboflow Flags

In Roboflow:
1. Open workspace `test-7vyqo`, project `ir-kibble`.
2. Filter by the monthly batch, for example `flagged-2026-05`.
3. Correct labels on uploaded frames.
4. Assign corrected images to the training split unless they are unusable.
5. Skip or delete frames from days where the camera was not pointing at the bowl.

Do not blindly accept every flagged frame. The point is to add corrected examples for recurring model mistakes, not to add noisy camera-position failures.

## 3. Optional: Mine More 24/7 Clips

Use `notebooks/batch_review.ipynb` only if the monthly Roboflow batch has too few useful corrections or misses an obvious failure mode.

Recommended settings for broad mining:

```python
FEEDING_WINDOW_ONLY = False
MAX_VIDEOS = 50
EXCLUDE_DATES = ["YYYYMMDD"]  # dates already reviewed or camera-misaligned days
```

Then run the notebook, upload flagged frames, and review the new batch in Roboflow before training.

## 4. Generate New Dataset Version

In Roboflow:
1. Generate a new dataset version, for example v15.
2. Export format: `yolov8`.
3. Keep preprocessing consistent with prior versions.

Before training, record:
- dataset version number,
- total image count,
- annotation count,
- per-class image/annotation counts.

## 5. Train

Use `notebooks/fair_feeder_v14.ipynb` as the training notebook and update v14 names to the new version.

For v15, update:

```python
version = project.version(15)
SAVE_DIR = Path("/kaggle/working/fair-feeder-v15")
name = "v15"
weights_dir = Path("runs/fair-feeder/v15/weights")
results_dir = Path("runs/fair-feeder/v15")
best_model = Path("runs/fair-feeder/v15/weights/best.pt")
```

Keep the baseline training settings unless the dataset balance clearly changed:

```python
YOLO("yolo11s.pt")
imgsz=[1280, 720]
rect=True
copy_paste=0.0
epochs=100
patience=20
```

Revisit `copy_paste` only after checking class counts. If Kibble is still the largest class, keep `copy_paste=0.0`.

## 6. Compare Models

For each model, collect:
- `best.pt` model file,
- `results.csv`,
- `results.png`,
- `confusion_matrix.png`,
- `confusion_matrix_normalized.png` if available,
- `PR_curve.png` if available,
- `F1_curve.png` if available,
- validation output from `model.val()`,
- Roboflow dataset version and class counts,
- notes about what new images were added.

Compare:
- overall mAP50,
- mAP50-95,
- per-class AP50 for Bowl, Dan, Dan_hand, Kibble, Sanbo,
- precision and recall per class,
- daily flag count trend after deployment.

Improvement-rate tracking:
- Absolute change: `new_metric - old_metric`.
- Relative change: `(new_metric - old_metric) / old_metric * 100`.
- Flag reduction: `(old_daily_flags - new_daily_flags) / old_daily_flags * 100`.

Example:

| Model | Dataset | mAP50 | mAP50-95 | Dan AP50 | Sanbo AP50 | Dan_hand AP50 | Kibble AP50 | Avg daily flags | Notes |
|-------|---------|-------|----------|----------|------------|---------------|-------------|-----------------|-------|
| v13 | Roboflow v13 | 0.956 | TBD | TBD | 0.959 | TBD | TBD | 10-20+ | Before data flywheel |
| v14 | Roboflow v14 | 0.957 | 0.754 | 0.936 | 0.985 | 0.936 | 0.931 | ~6 after first run | Added 231 corrected auto-flagged frames |
| v15 | Roboflow v15 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Monthly retrain |

Decision rule:
- Deploy if overall quality is stable or better and the target failure mode improves.
- Do not deploy if Dan, Sanbo, Dan_hand, or Kibble regresses in a way that affects daily compensation decisions.

## 7. Deploy

1. Upload `best.pt` to Google Drive model folder as `fair_feeder_vXX_yolov11s.pt`.
2. Update GitHub secret `GDRIVE_MODEL_FILE_ID` to the new Drive file ID.
3. Manually trigger `Fair Feeder Morning Report`.
4. Confirm Telegram uses the new model and the summary still looks correct.

## 8. Document

Update `docs/MODELS.md` with:
- model ID,
- mAP50,
- training date,
- notebook/code commit,
- Drive path,
- dataset size and what changed,
- per-class improvements/regressions.

Also update `tasks/todo.md` if this closes a retraining cycle.
