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
| Durable Delivery Ledger on Drive | `delivery_ledger_${DATE}_${CAMERA}.json` prevents duplicate deliveries across parallel/retry runs with item-level idempotency. |
| Deterministic CI DAG pipeline | `prepare` -> `tapo-report` -> `logitech-report` with target-date concurrency groups. |
| Cross-Camera Exclusion | Two cats cannot be in two rooms simultaneously; TAPO accepted FeedingTracker phase excludes Dan/Sanbo at Logitech. |
| Conflict Guard for Exclusion | When TAPO has identity conflict (contested phases, high conflict frames), cross-camera physical exclusion is disabled so incorrect identities are not forced onto the other camera. |
| Single Durable Authority (`delivery_registry.json`) | Single pre-created file in Drive updated via `update()` to avoid service account `403 storageQuotaExceeded` on `create()`, tracking house-level and camera-level completion. |
| Logitech Feeding Window Continuity (15s) | Pauses between dark RGB clips during feeding reach 11-15s due to recorder 5s timeout; window-aware grouping (05:55–06:35 AMS) bridges them while preserving separate visits. |
| Synchronized Side-by-Side Video | TAPO left, LOGITECH right, aligned to shared wall-clock time with neutral placeholders for gaps (no fake frames), 4x speedup, Telegram-safe <45 MB. |
| Rule D (VLM Failure Preservation) | Motion/differencing alone never forces Dan or Sanbo; cross-camera reconciliation requires VLM-proven cat presence. |

## Recent Failure Patterns

| Issue | Root cause | Fix |
|-------|------------|-----|
| Split breakfast feeding sessions (2026-09-05) | 11s motion pause in dark RGB split breakfast into 2 sessions. | 15s window-aware threshold (05:55–06:35 Amsterdam) unifies feeding pauses while preserving separate visits (>=47s). |
| False food theft alarm on contested TAPO (2026-09-05) | TAPO had 47% Dan / 53% Sanbo conflict; exclusion blindly trusted Sanbo at TAPO and forced Logitech to Dan. | Conflict Guard marks `has_conflict=True, exclusion_eligible=False` when TAPO is contested, suppressing exclusion. |
| Service account 403 on file create | GHA service account has zero storage quota in user Drive and cannot call `files().create()`. | Pre-created `delivery_registry.json` updated via Drive `files().update()`. |
| Duplicate morning deliveries (2026-09-01) | Multiple workflows ran concurrently without durable delivery state; partial runs resent all messages. | Durable Drive delivery ledger with preflight check and item-level HTTP 200 tracking. |
| Contradictory cat identity claims (2026-08-26) | Logitech visual VLM misclassified Sanbo as Dan while Dan was confirmed eating at Tapo. | Cross-camera physical exclusion reconciles identity using TAPO accepted feeding phases. |
| `feeding_log.csv` duplicates or wrong counts | Only last event was read; no same-day dedup. | Aggregate all `video_results`; remove today's row before append. |
| Scheduled action still starts after 8 AM | GitHub cron start time is not reliable. | Pi watchdog dispatches directly post-drain at 07:15 AMS; tertiary remote fallback at 08:00 AMS. |
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

