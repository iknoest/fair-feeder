"""
fix_cell_01_harden.py
Apply two hardening fixes to Cell 1 of morning_report.ipynb:
  A) Sort paths by name at the top of _group_by_gap
  B) Remove temp concat list file after successful ffmpeg merge
"""
import json

NB_PATH = 'C:/Users/AVAVAVA/.gemini/antigravity/scratch/fair-feeder/morning_report.ipynb'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][1]
src = ''.join(cell['source'])

# ── Fix A: sort paths by name at start of _group_by_gap ──
OLD_A = (
    "def _group_by_gap(paths, gap_sec=STITCH_GAP_SECONDS):\n"
    "    if not paths:\n"
    "        return []\n"
    "    timed = [(p, *_parse_clip_times(p.name)) for p in paths]"
)
NEW_A = (
    "def _group_by_gap(paths, gap_sec=STITCH_GAP_SECONDS):\n"
    "    if not paths:\n"
    "        return []\n"
    "    paths = sorted(paths, key=lambda p: p.name)\n"
    "    timed = [(p, *_parse_clip_times(p.name)) for p in paths]"
)

assert OLD_A in src, "Fix A: target string not found — notebook may have changed"
src = src.replace(OLD_A, NEW_A, 1)
print("Fix A applied: added sort in _group_by_gap")

# ── Fix B: remove concat list file after successful ffmpeg merge ──
OLD_B = (
    "                merged_sources[_mname] = [p.name for p in _group]\n"
    "                print(f\"  \u2705 Event {_gi+1}: merged {len(_group)} clips \u2192 {_mname}\")"
)
NEW_B = (
    "                merged_sources[_mname] = [p.name for p in _group]\n"
    "                _os2.remove(_list_file)\n"
    "                print(f\"  \u2705 Event {_gi+1}: merged {len(_group)} clips \u2192 {_mname}\")"
)

assert OLD_B in src, "Fix B: target string not found — notebook may have changed"
src = src.replace(OLD_B, NEW_B, 1)
print("Fix B applied: added _os2.remove(_list_file) after successful merge")

# Write back as list of lines (preserve Jupyter source format)
lines = src.split('\n')
new_source = [line + '\n' for line in lines[:-1]]
if lines[-1]:
    new_source.append(lines[-1])

cell['source'] = new_source

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("morning_report.ipynb updated successfully.")
