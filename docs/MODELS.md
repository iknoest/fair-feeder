# Model Versions — Fair Feeder

Trained YOLOv11 models for object detection (Dan, Sanbo, Dan_hand, Bowl, Kibble).
Each row documents one training run: model ID, performance metrics, training details, and storage location.

## Version History

| Model ID | mAP50 | Train Date | Colab Commit | Drive Path | Notes |
|----------|-------|-----------|--------------|-----------|-------|
| v13-final | 0.956 | 2026-01-25 | a419240 | `/My Drive/fair-feeder-models/yolov11s_v13_final.pt` | Dan_hand tuned; tested on 2 real videos |
| v14 | 0.957 | 2026-03-28 | — | `/My Drive/Fun Project/Cat monitor/model/fair_feeder_v14_yolov11s.pt` | Data flywheel retrain: 775 images (v13 + 231 auto-flagged). Sanbo AP50 0.959→0.985, Dan_hand precision→1.0. copy_paste=0.0 (kibble already largest class) |

## How to add a new model

1. Train in `fair_feeder_v14.ipynb` on Colab
2. After training, note down:
   - Model name (e.g., `v13-1` or `v14-improved`)
   - Final mAP50 (from `model.val()` output)
   - Training date
   - Colab commit hash (`git log --oneline -1`)
   - Drive path where model was saved (e.g., `/My Drive/fair-feeder-models/yolov11s_v13_1.pt`)
   - Any notes (e.g., "Added 50 new kibble examples", "Increased batch size to 32")
3. Add row to table above
4. Commit and push: `git add MODELS.md && git commit -m "docs: add model vX.Y"`

## Quick access

Use the latest **v14** model by default in all pipelines.
To use an older model, update the `MODEL_PATH` in the Download/Load cell:

```python
MODEL_PATH = '/content/drive/MyDrive/Fun Project/Cat monitor/model/fair_feeder_v14_yolov11s.pt'
```

## Metrics explanation

- **mAP50**: Mean Average Precision at IoU=0.50. Target: ≥ 0.85 for production.
- **Per-class AP50**: Check `model.val()` output for Dan, Sanbo, Dan_hand, Bowl, Kibble separately.
  Dan_hand is the most critical (hand-feeding detection).
