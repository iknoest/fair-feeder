# Morning Report Pipeline

`morning_report.ipynb` is the production report pipeline. It runs in CI through
papermill and interactively in Colab.

## Pipeline Stages

| Phase | Cell ID | Purpose | Speed |
|-------|---------|---------|-------|
| 1 | `detect-and-cache` | YOLO inference, JPEG frame cache, annotated video | minutes |
| 2 | `analyze-from-cache` | FeedingTracker analysis from cache, no video I/O | under 2s |
| 2.5 | `auto-flag` | Low-confidence, blip, and conflict scan | under 1s |
| 2.6 | `roboflow-upload` | Upload flagged frames with tags | under 10s |
| 3 | `output-and-telegram` | Summaries, snapshots, timeline, Telegram | under 5s |

## Cache Format

The detection cache is a pickle with:
- `frames[i].detections` - YOLO boxes and metadata.
- `frames[i].timestamp` - OCR timestamp string.
- `frames[i].jpeg` - compressed JPEG bytes, about 50 KB/frame.

Tuning workflow:
1. Run Phase 1 once to create `_detections.pkl`.
2. Change thresholds in the Config cell.
3. Re-run Phase 2 only.
4. Re-run Phase 3 to save and send output.

## Analysis Rules

- Stitch clips only if the gap is 10 seconds or less.
- Larger gaps are separate feeding events.
- Each distinct event gets its own FeedingTracker analysis and Telegram block.
- Episode numbers are continuous across clips with a day-wide offset.
- `dan_hand_epN` and `kibble_dispensed_epN` use day-wide episode numbers.
- Phase-based eating attribution avoids double-counting shared eating periods.
- `peak_kibble = max(counts)` is the double-counting guard.
- Rolling median window is 3 for kibble smoothing.
- Dan_hand requires Dan body co-detection in the same frame.

## Snapshot Rules

Kibble snapshots are high-risk. Preserve these rules:

- Prefer a stable no-cat frame before cats cover the bowl.
- Do not lock `kibble_dispensed` at the Dan_hand falling edge.
- Keep collecting clean post-hand/pre-arrival frames until cat arrival or timeout.
- If no clean pre-cat frame exists, wait for 3 consecutive clear frames after
  Dan_hand.
- Final fallback is the highest-kibble frame within 5 seconds.
- Kibble snapshots must not depend on confirmed `Dan_hand`; if Dan_hand is absent
  or filtered out, still emit a generic `kibble_dispensed` snapshot at first bowl
  arrival using the best clean pre-cat frame.
- Do not send `kibble_start` snapshots. First visible Kibble frames can be noisy
  early frames that look like the old broken Telegram symptom.

## Telegram Output

- The first line is the action verdict because it appears in Telegram push
  notifications.
- Current verdict meanings:
  - Dan finished breakfast.
  - Give Dan about N kibble.
  - Feeding machine may not be working.
- Do not include schedule delay or explanatory `Why:` lines in Telegram.
- Keep scheduler heartbeat in logs, GitHub summaries, and `feeding_log.csv`.
- Annotated video must be boxes-only; labels and percentages obscure kibble.

## Bowl Position Alert

- Check full bowl visibility, not frame center.
- The bowl normally sits on the right side of the Tapo frame.
- Alert only when the bowl is missing or its box is clipped/not fully visible.
- A center around 78%, 61% can be a good position.

