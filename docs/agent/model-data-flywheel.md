# Model and Data Flywheel

## Current Status

Stage: production pipeline running. V14 is deployed. V15 is a candidate and must
be validated against V14 on a fixed holdout before deployment.

Active model notes:
- V14 deployed baseline: historical mAP50 0.957; fresh smoketest rerun mAP50
  0.690.
- V15 candidate: trained from 155 manually revised April flagged images; fresh
  standalone/smoketest-style mAP50 0.741.
- Deployment decision requires fixed V14/V15 validation comparison.

## Data Flywheel State

Phase C:
- C1 auto-flag and Roboflow upload verified in CI on 2026-03-26.
- C2 batch reprocessing uploaded 231 frames with pre-annotations.
- C3 V14 trained on 2026-03-28 with 775 images.
- V14 deployed to CI by updating `GDRIVE_MODEL_FILE_ID`.
- V15 decision still pending fixed validation.

## Training Decisions

- YOLOv11s is preferred over m/l because the dataset is small; small-object
  performance comes from 1280px input more than model size.
- 1280px inference is required because kibble is tiny.
- `rect=True` preserves the Tapo 16:9 aspect ratio.
- MixUp augmentation is off because it destroys small kibble detail.
- Vertical flip augmentation is off because the camera is fixed overhead.
- Copy-paste augmentation is off for V14; Kibble already dominates annotations,
  while Sanbo and Dan are the data bottlenecks.

## Roboflow Workflow

- Use pre-annotations with `is_prediction=True`; the reviewer corrects model
  output instead of labeling from scratch.
- Use monthly batches named `flagged-YYYY-MM`.
- Use `roboflow_uploaded.txt` on Drive for deduplication.
- Reference mAP50 and per-class AP50 when discussing model quality.

## Planned Later

- Bowl ROI zone filter in `motion_recorder.py`.
- Lightweight Dan/Sanbo classifier on Pi to tag clip filenames.
- Telegram-interactive flagging by reply-to-flag.

