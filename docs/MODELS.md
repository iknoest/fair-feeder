# Model Versions - Fair Feeder

Trained YOLOv11 models for object detection (Dan, Sanbo, Dan_hand, Bowl, Kibble).
Each row documents one training run: model ID, performance metrics, training details, and storage location.

## Version History

| Model ID | mAP50 | mAP50-95 | Train Date | Colab Commit | Drive Path | Notes |
|----------|-------|----------|------------|--------------|------------|-------|
| v13-final | 0.956 baseline; 0.421 in later pasted standalone val | 0.739 baseline; 0.360 in later pasted standalone val | 2026-01-25 | a419240 | `/My Drive/fair-feeder-models/yolov11s_v13_final.pt` | Dan_hand tuned; tested on 2 real videos. Later V13 pasted validation used only 54 validation images, so do not compare directly with the original baseline without rerunning a fixed validation command. |
| v14 | 0.957 historical; 0.690 in 2026-05-02 smoketest rerun | 0.754 historical; 0.564 in 2026-05-02 smoketest rerun | 2026-03-28 | TBD | `/My Drive/Fun Project/Cat monitor/model/fair_feeder_v14_yolov11s.pt` | Data flywheel retrain: 775 images (v13 + 231 auto-flagged). First deployed run: ~6 flags vs 10-20+ on v13. Keep historical metrics separate from fresh smoketest validation on `/content/IR-kibble-14/valid` with 109 images. |
| v15-candidate | 0.741 pasted standalone val; training run peaked near 0.949 | 0.594 pasted standalone val; training run peaked near 0.743 | 2026-05-01 | TBD | `C:\Users\AVAVAVA\Downloads\fair_feeder_v15_yolov11s.pt` | Candidate trained after adding 155 manually revised April flagged images. Better than the v14 smoketest rerun overall, but v14/v15 still use different validation splits. For deployment, prefer daily report quality plus a fixed holdout validation set. |

## How To Add A New Model

1. Train in `fair_feeder_v14.ipynb` on Colab/Kaggle.
2. After training, note down:
   - Model name, for example `v16-candidate`.
   - Final mAP50 and mAP50-95 from the fixed validation command.
   - Training date.
   - Notebook/code commit hash.
   - Drive path where model was saved.
   - Dataset version, image count, annotation count, and what changed.
3. Add a row to the table above.
4. Commit and push: `git add docs/MODELS.md && git commit -m "docs: add model vX"`.

## Quick Access

Use the latest deployed **v14** model by default in all pipelines until V15 is validated and deployed.

To use the deployed model, update the `MODEL_PATH` in the Download/Load cell:

```python
MODEL_PATH = "/content/drive/MyDrive/Fun Project/Cat monitor/model/fair_feeder_v14_yolov11s.pt"
```

For the monthly decision workflow, use [model-improvement-handbook.md](model-improvement-handbook.md).

## Metrics Explanation

- **mAP50**: Mean Average Precision at IoU=0.50. Target: >= 0.85 for production.
- **mAP50-95**: Stricter localization metric averaged across IoU thresholds. Use it to catch box-quality regressions.
- **Per-class AP50**: Check `model.val()` output for Dan, Sanbo, Dan_hand, Bowl, Kibble separately.
- **Daily flag trend**: The final deployment check. A model that looks good in validation but increases bad Telegram decisions should not be promoted.
