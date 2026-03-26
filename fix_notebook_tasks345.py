"""Patch morning_report.ipynb for Tasks 3, 4, and 5.

Task 3: Insert Phase 2.5 (auto-flagging) and Phase 2.6 (Roboflow upload) cells after Phase 2.
Task 4: Append flag_summary_text to Telegram message in Phase 3 cell.
Task 5: Remove Drive video upload block from Cell 14 (CSV cell).
"""
import json
import sys
import copy

NB_PATH = 'morning_report.ipynb'

# ── Read notebook ──────────────────────────────────────────────
with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def clean(s):
    """Strip Windows CRLF from string."""
    return s.replace('\r', '')


def make_code_cell(source_str):
    """Create a new code cell dict with the given source."""
    lines = clean(source_str).split('\n')
    # Convert to line-per-entry format with trailing \n except last line
    source_list = [line + '\n' for line in lines[:-1]]
    if lines[-1]:  # last line has no trailing newline
        source_list.append(lines[-1])
    else:
        # source ended with \n, last entry is empty string — skip it
        pass
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_list,
    }


# ── Task 3: Find Phase 2 cell and insert Phase 2.5 + 2.6 after it ──
phase2_idx = None
for i, cell in enumerate(cells):
    src = ''.join(cell['source'])
    if 'Phase 2' in src and 'FeedingTracker' in src and 'summarize' in src:
        phase2_idx = i
        break

if phase2_idx is None:
    print("ERROR: Could not find Phase 2 analytics cell")
    sys.exit(1)

print(f"Task 3: Found Phase 2 cell at index {phase2_idx}")

PHASE_25_SOURCE = """\
# Phase 2.5 — Auto-flag suspicious detections
from flagging import flag_detections
import pickle

FLAG_CONF_THRESHOLD = 0.40
FLAG_BLIP_MAX_FRAMES = 2
FLAG_BLIP_GAP_FRAMES = 5
FLAG_IOU_CONFLICT = 0.50
FLAG_KIBBLE_JUMP = 5
FLAG_DEDUP_WINDOW = 3

all_flagged = {}

for vr in video_results:
    vid_stem = vr['vid_stem']
    cache_path = os.path.join(OUTPUT_DIR, f"{vid_stem}_detections.pkl")
    if not os.path.exists(cache_path):
        print(f"  [skip] No cache for {vid_stem}")
        continue

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    flagged = flag_detections(
        cache['frames'],
        conf_threshold=FLAG_CONF_THRESHOLD,
        blip_max_frames=FLAG_BLIP_MAX_FRAMES,
        blip_gap_frames=FLAG_BLIP_GAP_FRAMES,
        iou_conflict=FLAG_IOU_CONFLICT,
        kibble_jump=FLAG_KIBBLE_JUMP,
        dedup_window=FLAG_DEDUP_WINDOW,
    )
    all_flagged[vid_stem] = flagged
    print(f"  {vid_stem}: {len(flagged)} frames flagged")

total_flagged = sum(len(v) for v in all_flagged.values())
print(f"\\nTotal flagged: {total_flagged} frames")"""

PHASE_26_SOURCE = """\
# Phase 2.6 — Upload flagged frames to Roboflow
from roboflow_upload import upload_flagged_frames, format_telegram_flag_summary, UploadResult

ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', '')
ROBOFLOW_WORKSPACE = os.environ.get('ROBOFLOW_WORKSPACE', '')
ROBOFLOW_PROJECT = 'ir-kibble'

combined_result = UploadResult()

if not ROBOFLOW_API_KEY:
    print("ROBOFLOW_API_KEY not set — skipping Roboflow upload")
    flag_summary_text = "Roboflow upload skipped (no API key)"
elif total_flagged == 0:
    print("No frames flagged — nothing to upload")
    flag_summary_text = format_telegram_flag_summary(combined_result)
else:
    for vid_stem, flagged in all_flagged.items():
        if not flagged:
            continue
        print(f"  Uploading {len(flagged)} frames from {vid_stem}...")
        result = upload_flagged_frames(
            flagged,
            api_key=ROBOFLOW_API_KEY,
            workspace=ROBOFLOW_WORKSPACE,
            project=ROBOFLOW_PROJECT,
            video_stem=vid_stem,
        )
        combined_result.uploaded += result.uploaded
        combined_result.failed += result.failed
        for tag, count in result.tag_counts.items():
            combined_result.tag_counts[tag] = combined_result.tag_counts.get(tag, 0) + count

    flag_summary_text = format_telegram_flag_summary(combined_result)
    print(f"\\n{flag_summary_text}")"""

cell_25 = make_code_cell(PHASE_25_SOURCE)
cell_26 = make_code_cell(PHASE_26_SOURCE)

# Insert after Phase 2 cell
cells.insert(phase2_idx + 1, cell_25)
cells.insert(phase2_idx + 2, cell_26)

print(f"  Inserted Phase 2.5 at index {phase2_idx + 1}")
print(f"  Inserted Phase 2.6 at index {phase2_idx + 2}")

# ── Task 4: Find Phase 3 cell and inject flag summary before send_telegram ──
# Phase 3 cell is now shifted by 2 indices
phase3_idx = None
for i, cell in enumerate(cells):
    src = ''.join(cell['source'])
    if 'Phase 3' in src and 'send_telegram' in src:
        phase3_idx = i
        break

if phase3_idx is None:
    print("ERROR: Could not find Phase 3 cell")
    sys.exit(1)

print(f"Task 4: Found Phase 3 cell at index {phase3_idx}")

# Find the send_telegram_summary call and insert flag summary before it
phase3_source = ''.join(cells[phase3_idx]['source'])

inject_before = '        send_telegram_summary('
inject_code = """\
        # Append flag summary to Telegram message
        if 'flag_summary_text' in dir():
            summary_text += '\\n\\n' + flag_summary_text

"""

if inject_before not in phase3_source:
    print(f"ERROR: Could not find '{inject_before.strip()}' in Phase 3 cell")
    sys.exit(1)

phase3_source = phase3_source.replace(inject_before, clean(inject_code) + inject_before)

# Rebuild source list
phase3_lines = phase3_source.split('\n')
phase3_source_list = [line + '\n' for line in phase3_lines[:-1]]
if phase3_lines[-1]:
    phase3_source_list.append(phase3_lines[-1])
cells[phase3_idx]['source'] = phase3_source_list
print(f"  Injected flag_summary_text append before send_telegram_summary()")

# ── Task 5: Remove Drive video upload block from CSV cell ──
csv_idx = None
for i, cell in enumerate(cells):
    src = ''.join(cell['source'])
    if 'feeding_log.csv' in src and 'DictWriter' in src:
        csv_idx = i
        break

if csv_idx is None:
    print("ERROR: Could not find CSV cell")
    sys.exit(1)

print(f"Task 5: Found CSV cell at index {csv_idx}")

csv_source = ''.join(cells[csv_idx]['source'])

# Find and remove the video upload block
upload_start = "    # ── Annotated video upload"
upload_end_marker = "                print(f'⚠️ Annotated video not found: {_out_vid}')"

start_pos = csv_source.find(upload_start)
if start_pos == -1:
    # Try alternative marker
    upload_start = "    # ── Annotated video upload ────────────────────────────────────"
    start_pos = csv_source.find(upload_start)

if start_pos == -1:
    print("ERROR: Could not find video upload block start in CSV cell")
    sys.exit(1)

end_pos = csv_source.find(upload_end_marker)
if end_pos == -1:
    print("ERROR: Could not find video upload block end in CSV cell")
    sys.exit(1)

# Include the end marker line plus its newline
end_pos = csv_source.find('\n', end_pos) + 1

replacement = clean("""\
    # Drive video upload removed — SA has zero storage quota (Issue #33)
    # Annotated videos are delivered via Telegram; archive via Colab.
""")

csv_source = csv_source[:start_pos] + replacement + csv_source[end_pos:]

# Rebuild source list
csv_lines = csv_source.split('\n')
csv_source_list = [line + '\n' for line in csv_lines[:-1]]
if csv_lines[-1]:
    csv_source_list.append(csv_lines[-1])
cells[csv_idx]['source'] = csv_source_list
print(f"  Removed Drive video upload block, replaced with comment")

# ── Save notebook ──────────────────────────────────────────────
nb['cells'] = cells
with open(NB_PATH, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print(f"\nSaved {NB_PATH}")

# ── Verification ───────────────────────────────────────────────
print("\n=== VERIFICATION ===")
with open(NB_PATH, encoding='utf-8') as f:
    nb2 = json.load(f)

for i, cell in enumerate(nb2['cells']):
    first_line = cell['source'][0].strip() if cell['source'] else '(empty)'
    print(f"  Cell {i} ({cell['cell_type']}): {first_line[:100]}")
