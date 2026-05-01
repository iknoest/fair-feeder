# Model Improvement Handbook

Use this handbook at the start of each month, or whenever Telegram reports and Roboflow flags show a repeated failure mode. The default decision is not always "retrain"; choose between monitoring, targeted fine-tuning, or a full retrain based on evidence.

## 1. Monthly Decision

Start with the last 3-4 weeks of evidence:

- `feeding_log.csv`: `flagged_frames`, `roboflow_uploaded`, `roboflow_skipped`, `roboflow_failed`, `flag_top_tags`.
- Telegram reports: wrong compensation verdicts, wrong cat share, missing hand-feeding, bad kibble start/end counts.
- Roboflow monthly batch: how many flagged images are actually useful after review.
- Camera-position alerts: exclude camera-misaligned days from model-quality judgment.

Decision guide:

| Situation | Action |
|-----------|--------|
| Daily reports are correct and useful corrected images are low | Do not retrain; keep monitoring |
| One failure mode repeats and there are ~50-150 useful corrected images | Fine-tune from the current best model on the full dataset plus new images |
| Camera/lighting changed, or reports are wrong in core decisions | Full retrain on the latest full dataset |
| Validation improves but daily compensation gets worse | Do not deploy; debug thresholds/report logic first |

## 2. Review Model Drift

Use monthly flag trends to decide what data is needed.

Direction guide:

- `conflict-dan-sanbo`: improve Dan/Sanbo identity and overlap examples.
- `low-conf-dan` or `low-conf-sanbo`: add varied cat pose, distance, and lighting examples.
- `blip-dan`, `blip-sanbo`, `blip-dan_hand`: check one-frame hallucinations.
- `no-codetect-dan_hand`: correct hand-feeding labels and ensure Dan body is also labeled.
- `kibble-jump-*`: only high-value when the bowl is clear; skip normal cat-eating occlusion.
- camera-position alerts: exclude bad camera days or label only if the bowl is visible and useful.

## 3. Review Roboflow Flags

In Roboflow:

1. Open workspace `test-7vyqo`, project `ir-kibble`.
2. Filter by the monthly batch, for example `flagged-2026-05`.
3. Correct labels on uploaded frames.
4. Assign corrected images to the training split unless they are unusable.
5. Skip or delete frames from days where the camera was not pointing at the bowl.

Do not blindly accept every flagged frame. The point is to add corrected examples for recurring model mistakes, not to add noisy camera-position failures.

## 4. Optional: Mine More 24/7 Clips

Use `notebooks/batch_review.ipynb` only if the monthly Roboflow batch has too few useful corrections or misses an obvious failure mode.

Recommended settings for broad mining:

```python
FEEDING_WINDOW_ONLY = False
MAX_VIDEOS = 50
EXCLUDE_DATES = ["YYYYMMDD"]  # dates already reviewed or camera-misaligned days
```

Then run the notebook, upload flagged frames, and review the new batch in Roboflow before training.

## 5. Generate Dataset Version

In Roboflow:

1. Generate a new dataset version, for example v16.
2. Export format: `yolov8`.
3. Keep preprocessing consistent with prior versions.

Before training, record:

- dataset version number,
- total image count,
- annotation count,
- per-class image/annotation counts,
- how many manually corrected monthly flagged images were added,
- excluded dates and why.

## 6. Train or Fine-Tune

Use `notebooks/fair_feeder_v14.ipynb` as the training notebook template and update version names.

For a normal full retrain:

```python
YOLO("yolo11s.pt")
imgsz=[1280, 720]
rect=True
copy_paste=0.0
epochs=100
patience=20
```

For a monthly fine-tune, start from the current best Fair Feeder model but still train on the full dataset. Do not train only on the new month of images, because that risks forgetting older lighting, pose, and feeding patterns.

Example:

```python
YOLO("/kaggle/input/fair-feeder-models/fair_feeder_v15_yolov11s.pt")
imgsz=[1280, 720]
rect=True
copy_paste=0.0
epochs=25
patience=8
```

Revisit `copy_paste` only after checking class counts. If Kibble is still the largest class, keep `copy_paste=0.0`.

## 7. Compare Models

Always compare with one fixed validation command and one fixed validation dataset. Do not mix the training-run `results.csv` best epoch with a separate `model.val()` command unless the command, weights, image size, and dataset version are recorded.

Collect for each model:

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

Improvement-rate formulas:

- Absolute change: `new_metric - old_metric`.
- Relative change: `(new_metric - old_metric) / old_metric * 100`.
- Flag reduction: `(old_daily_flags - new_daily_flags) / old_daily_flags * 100`.

### V13 -> V14 -> V15 Snapshot

The V13/V14 values below come from the historical project notes. The V15 values come from the May 2026 validation output after adding 155 manually revised April flagged images.

| Model | Dataset | Images | mAP50 | mAP50-95 | Precision | Recall | Notes |
|-------|---------|--------|-------|----------|-----------|--------|-------|
| v13 | Roboflow v13 | 54 validation images in pasted run | 0.421 in pasted validation; 0.956 in earlier project baseline | 0.360 in pasted validation; 0.739 in earlier baseline | 0.524 | 0.503 | The pasted V13 run appears to use a small validation set and is not directly comparable with the old baseline row. |
| v14 | Roboflow v14 | 775 training images | 0.957 | 0.754 | 0.941 | 0.897 | Added 231 corrected auto-flagged frames. First deployed run dropped flags from 10-20+ to ~6 and removed `blip-sanbo`. |
| v15 | Roboflow v15 | 139 validation images | 0.741 in pasted standalone validation | 0.594 | 0.815 | 0.780 | Added 155 manually revised April flagged images. Strong improvement versus the pasted V13 validation, but not directly comparable with V14 unless run on the same validation setup. |

V15 versus pasted V13 validation:

| Metric | V13 | V15 | Absolute change | Relative change |
|--------|-----|-----|-----------------|-----------------|
| mAP50 | 0.421 | 0.741 | +0.320 | +76.0% |
| mAP50-95 | 0.360 | 0.594 | +0.234 | +65.0% |
| Precision | 0.524 | 0.815 | +0.291 | +55.5% |
| Recall | 0.503 | 0.780 | +0.277 | +55.1% |

Per-class AP50 snapshot:

| Class | V13 pasted val | V14 historical | V15 pasted val | Read |
|-------|----------------|----------------|----------------|------|
| Bowl | 0.349 | 0.995 | 0.688 | V15 is much better than pasted V13, but below the V14 historical validation. |
| Dan | 0.522 | 0.936 | 0.804 | Good recovery versus pasted V13; still needs fixed V14/V15 validation before deployment. |
| Dan_hand | 0.113 | 0.936 | 0.606 | Largest practical gap; keep collecting hand-feeding examples. |
| Kibble | 0.354 | 0.931 | 0.772 | Better than pasted V13, still sensitive to occlusion and tiny-object noise. |
| Sanbo | 0.769 | 0.985 | 0.837 | Production-usable in V15 pasted val, but V14 historical remains stronger. |

V15 caveat: the uploaded `results.csv`/`results.png` show training-run validation peaking around `mAP50=0.949` and `mAP50-95=0.743` near epoch 41, while the pasted standalone validation shows `mAP50=0.741` and `mAP50-95=0.594`. Treat this as an artifact mismatch until the exact validation command and model weights are confirmed.

### V15 Performance Notes

- `Sanbo` is strong enough for production decisions.
- `Dan` improved materially and should be monitored mainly during overlap with Sanbo.
- `Dan_hand` improved but remains a priority because hand-feeding controls the report interpretation.
- `Kibble` remains the hardest class because visible pieces are small and cats frequently occlude the bowl.
- `Bowl` is useful for report analysis, but the 24/7 Pi camera-position alert currently uses YOLOv8n COCO `bowl`, not this custom model.

## 8. Deploy

Deploy only when the model improves the actual daily decision quality.

1. Upload `best.pt` to Google Drive model folder as `fair_feeder_vXX_yolov11s.pt`.
2. Update GitHub secret `GDRIVE_MODEL_FILE_ID` to the new Drive file ID.
3. Manually trigger `Fair Feeder Morning Report`.
4. Confirm Telegram uses the new model and the summary still looks correct.
5. Monitor daily flag counts for at least 7 days before judging the deployment successful.

Do not deploy if Dan, Sanbo, Dan_hand, or Kibble regresses in a way that affects compensation decisions.

## 9. Monthly Maintenance Guideline

At month end:

1. Export `feeding_log.csv` and summarize daily flags by tag.
2. Review the Roboflow monthly batch and count only useful corrected images.
3. Decide: monitor, fine-tune, or full retrain.
4. If training, run a fixed validation command for the old and new model on the same dataset.
5. Update `docs/MODELS.md` with the model row and the comparison note.
6. Update this handbook if the decision rule changes.
7. Update `tasks/todo.md` only with real remaining work.

Recommended May focus:

- Add hard `Dan_hand` examples: partial hand, hand near bowl, hand occluding Dan.
- Keep only meaningful `kibble-jump-*` frames where the bowl is clear.
- Add empty bowl, shiny bowl, partial bowl, and camera-shift examples.
- Add Dan/Sanbo overlap frames where identity confusion affects the kibble share.
- Exclude camera-misaligned days from training unless the bowl is still visible and correctly labeled.
