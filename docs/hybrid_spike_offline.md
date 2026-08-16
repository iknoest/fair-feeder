# Hybrid Spike Offline Runbook

This document describes how to execute the Phase 1A offline spike to evaluate the VLM approach for Logitech frames and verify the Tapo frozen model.

## Prerequisites
1. Local copies of historical feeding clips.
   - For current validation, use recent Logitech clips from the actual current setup.
   - Include the lowest-light Logitech clips available from the last month if possible, but do not block if they are not dark. Future winter Logitech darkness should be logged as a seasonal risk, not solved in this scaffold.
   - Use Dec-Jan Tapo dark IR clips separately to stress-check the IR/Tapo boundary and verify that v14 remains trustworthy in-domain.
   - If you do not have these clips locally, you must first download them from Google Drive.

## Execution
Run the extraction script from the repository root:

To process both cameras (example):
```bash
python scripts/hybrid_spike_offline.py \
    --tapo-dir /path/to/tapo/clips \
    --logitech-dir /path/to/logitech/clips \
    --out-dir .agent/artifacts/hybrid_spike \
    --max-tapo-windows 3 \
    --max-logitech-windows 5
```

Partial runs are also supported. To process only Logitech:
```bash
python scripts/hybrid_spike_offline.py --logitech-dir /path/to/logitech/clips
```

To process only Tapo:
```bash
python scripts/hybrid_spike_offline.py --tapo-dir /path/to/tapo/clips
```

## Outputs
The script will generate the following in the `--out-dir`:
1. `manifest.csv`: A metadata table of all extracted keyframes with brightness/sharpness scores.
2. `summary.json`: A summary metrics JSON file.
3. `logitech_contact_sheet.jpg` (if Logitech processed): A visual grid of the extracted Logitech frames.
4. `tapo_contact_sheet.jpg` (if Tapo processed): A visual grid of the extracted Tapo frames.
5. `vlm_prompt_template.md`: The prompt template to use when manually validating the VLM capability.

## Next Steps
- Open the contact sheets to see if the frames are visually decipherable by a human.
- Provide the Logitech keyframes to a VLM (Gemini/Claude) using the generated `vlm_prompt_template.md` and compare the output to human ground truth.
