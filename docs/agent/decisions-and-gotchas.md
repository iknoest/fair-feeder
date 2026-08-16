# Decisions and Gotchas

This file preserves project-specific lessons that are too large for root context.

## Active Design Decisions

| Decision | Reason |
|----------|--------|
| YOLOv11s, not m/l | Dataset is too small for larger models; 1280px input carries small-object detection. |
| 1280px inference | 640px misses too many kibble detections. |
| `rect=True` everywhere | Preserves Tapo 16:9 aspect ratio. |
| Phase-based attribution | Prevents double-counting during overlapping feeding. |
| Rolling median window 3 | Removes single-frame Kibble flicker without hiding real transitions. |
| Dan_hand requires Dan body | Eliminates stray hand false positives. |
| `peak_kibble = max(counts)` | `first_clear` underestimates starting kibble when cats occlude the bowl. |
| Boxes-only annotated video | Labels and percentages obscure Kibble. |
| Stitch only gaps <= 10s | Larger gaps are separate feeding events. |
| Per-event Telegram block | Each feeding event gets its own report. |
| `_fmt_time()` strips same-day date | Reduces mobile Telegram clutter. |
| Verdict first in Telegram | Push notifications need the action immediately. |
| Continuous episode numbers | Snapshot keys are day-wide, not per-clip. |
| Pre-cat `kibble_dispensed` | Telegram needs inspectable Kibble before cats cover the bowl. |
| No `kibble_start` snapshot | Early first-visible Kibble frames are noisy and misleading. |
| Bowl visibility alert | Alert on missing/clipped bowl, not off-center placement. |
| Compensation equals Sanbo Kibble eaten | Directly answers how much extra Dan needs. |
| ffmpeg crf=28, 720p | Keeps most Telegram videos inline under size limits. |
| `RUNNING_IN_CI` guard | Cleaner than try/except around Colab-only code. |
| `tqdm.auto` | Works in CI and notebooks. |
| Drive `update()`, not `create()` | Service account has no personal Drive quota. |
| CI is cron only | Interactive tuning and archives belong in Colab. |
| Roboflow pre-annotations | Review is faster than labeling from scratch. |
| Monthly Roboflow batches | About 12 batches/year is manageable. |
| Drive upload dedup file | `roboflow_uploaded.txt` avoids duplicate uploads. |
| Tapo OCR replacement order | `\|:` must become `:1` before `\|` becomes `1`. |
| No MixUp or vertical flip | Protects small Kibble detail and fixed camera geometry. |
| `last_motion_time` stop timer | Tapo ONVIF motion events arrive in bursts with gaps. |
| Independent sync folders per camera | Prevents rclone collision and simplifies cleanup/analytics. |
| Automatic rclone Folder ID detection | Uses `--drive-root-folder-id` if `RCLONE_DEST_PATH` looks like an ID. |

## Recent Failure Patterns

| Issue | Root cause | Fix |
|-------|------------|-----|
| `feeding_log.csv` duplicates or wrong counts | Only last event was read; no same-day dedup. | Aggregate all `video_results`; remove today's row before append. |
| Scheduled action still starts after 8 AM | GitHub cron start time is not reliable; 2026-05-18 and 2026-05-19 were delayed about 4h10m. | Schedule early at `23 0 * * *` UTC, avoid minute 0, raise timeout to 360, and wait until 06:35 Amsterdam if needed. |
| Annotated video missing from Drive in CI | Service account `create()` hit zero quota. | Do not upload large binaries from CI; Colab archives. |
| Telegram sent unmerged short clip | Phase 1/2 rescanned `SOURCE_DIR`. | Guard rescan behind `if not RUNNING_IN_CI:`. |
| Report said 0 Kibble despite timeline Kibble | Clear-count logic only searched no-cat frames. | Add phase-entry/exit fallback methods. |
| Morning captures missed | Naive UTC date filtering. | Use Europe/Amsterdam timezone. |
| Papermill `IndentationError` | Windows `\r\n` leaked into notebook cell source. | Strip `\r` when writing notebook JSON. |
| `tqdm.notebook` crashed in CI | Widget server absent in papermill. | Use `tqdm.auto`. |
| Empty secret CI runs | Secret existed in settings but was not listed under workflow `env:`. | List every consumed secret under the step. |
| Recording stopped during continuous motion | ONVIF events arrive in bursts with 1-3s gaps. | Use `last_motion_time`, stop after 5s quiet. |
| Overlapping phases double-counted Kibble | Per-episode accounting. | Phase-based attribution and peak guard. |
| Dan_hand false positives | No Dan body co-detection requirement. | Require Dan body in same frame. |
| Help command showing old text | Duplicate method definition at end of file overrode updates. | Delete redundant methods when cleaning up or refactoring classes. |

