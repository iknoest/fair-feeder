# Phase 2A: Logitech VLM Shadow

This document describes the offline VLM shadow script for Logitech cameras.

## Background
Phase 1D diagnostics proved that the current frozen `fair_feeder_v14_yolov11s.pt` model (trained only on Tapo IR data) is unreliable for the top-down RGB Logitech camera. Retraining YOLO is not an option right now due to maintenance constraints. Therefore, the Logitech stream will pivot to a Vision-Language Model (VLM) for cat identity and bowl state inference.

## The Script: `scripts/logitech_vlm_shadow.py`

This script is a shadow scaffold:
- It processes Logitech production clips from Google Drive.
- It extracts representative frames where motion is detected.
- It generates the exact VLM prompt, schema, contact sheet, summary, and CSV manifest.
- **It does NOT change production code (`morning_report.ipynb`) or upload/mutate any data.**

### Prepare-only mode (Default)
This script runs strictly as a prepare-only tool. It generates the necessary artifacts (images, prompts) to simulate the VLM analysis offline.
```bash
python scripts/logitech_vlm_shadow.py --date 2026-07-04 --out-dir .agent/artifacts/logitech_vlm_shadow_20260704
```

### Shadow-run mode
The `--run-vlm` flag is intentionally blocked until real API integration is implemented. If you pass this flag, the script will gracefully exit after generating artifacts, outputting:
`[STOP] Real VLM API execution is not implemented in this scaffold yet. Use prepare-only artifacts for manual VLM testing.`

## Manual Next Steps
Until the automated VLM integration is fully complete, use the prepare-only mode. Take the generated contact sheet or selected frames from `logitech_vlm_frames/` and supply them to a VLM (such as ChatGPT/Claude/Gemini) alongside the text generated in `logitech_vlm_prompt.md` to manually verify the logic and sanity check the extraction.

## Report Wording Recommendation
If Logitech YOLO is unreliable and the VLM integration is not yet active in production, the morning report should **not** say "feeding machine not working" based on Logitech alone. It should output a fallback message such as:
> *“Logitech visual check inconclusive: cat/food may be visible, but current detector is not reliable for this camera.”*
